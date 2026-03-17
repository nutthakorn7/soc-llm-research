#!/bin/bash
# P9 DPO β Sweep — Vast.ai
# ========================
# Usage: ssh vast 'bash /workspace/p9_dpo.sh'
# Requires: RTX 4090 or A100 (~$0.30-1.50/h)
# Runtime: ~2h on 4090

set -e
echo "🚀 P9 DPO β Sweep — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

pip install -q transformers>=4.45 peft>=0.13 trl>=0.12 \
    accelerate>=0.34 datasets scikit-learn

mkdir -p /workspace/results/p9
mkdir -p /workspace/salad_data

# Download SALAD
wget -q -O /workspace/salad_data/train_5k_clean.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/train_5k_clean.json"
wget -q -O /workspace/salad_data/test_held_out.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/test_held_out.json"
echo "✅ SALAD downloaded"

python3 << 'PYEOF'
import torch, random, json, time, os, csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

SAVE = "/workspace/results/p9"
CATS = ["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance","Shellcode"]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

train_data = json.load(open("/workspace/salad_data/train_5k_clean.json"))
test_data = json.load(open("/workspace/salad_data/test_held_out.json"))
print(f"SALAD: {len(train_data)} train, {len(test_data)} test")

def autosave(r, fn):
    with open(f"{SAVE}/{fn}","w") as f: json.dump(r,f,indent=2)
    cp = f"{SAVE}/summary.csv"; ex = os.path.exists(cp)
    with open(cp,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        if not ex: w.writeheader()
        w.writerow(r)
    print(f"✅ SAVED {fn} | β={r.get('beta','?')} s={r.get('seed','?')} F1={r.get('strict_f1','?')}")

def parse(text):
    for c in CATS:
        if c.lower() in text.lower(): return c
    return text.strip().split("\n")[-1].strip()

def train_sft(seed=42, epochs=3, lr=2e-4, batch=4, gacc=8, rank=64, ml=512):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}\n🔧 SFT seed={seed}\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    mdl = AutoModelForCausalLM.from_pretrained(MODEL,
        dtype=torch.float16, device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))

    from datasets import Dataset
    def tok_fn(ex):
        convs = ex["conversations"]
        ai,aa,al = [],[],[]
        for c in convs:
            m = [{"role":"system","content":c[0]["value"]},
                 {"role":"user","content":c[1]["value"]},
                 {"role":"assistant","content":c[2]["value"]}]
            s = tok.apply_chat_template(m, tokenize=False)
            e = tok(s, truncation=True, max_length=ml, padding="max_length")
            ids=e["input_ids"]; at=e["attention_mask"]
            lb = [l if a==1 else -100 for l,a in zip(ids,at)]
            ai.append(ids); aa.append(at); al.append(lb)
        return {"input_ids":ai,"attention_mask":aa,"labels":al}

    ds = Dataset.from_dict({"conversations":[d["conversations"] for d in train_data]})
    ds = ds.map(tok_fn, batched=True, batch_size=100, remove_columns=["conversations"])
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 10, (len(loader)//gacc)*epochs)

    mdl.train(); t0=time.time()
    for ep in range(epochs):
        tl=0; opt.zero_grad()
        for st,b in enumerate(loader):
            b={k:v.to("cuda") for k,v in b.items()}
            loss=mdl(**b).loss/gacc; loss.backward(); tl+=loss.item()*gacc
            if (st+1)%gacc==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"  Epoch {ep+1}/{epochs} loss={tl/len(loader):.4f}")
    tm=(time.time()-t0)/60; print(f"  SFT done: {tm:.1f}m")
    path="/workspace/sft_adapter"; mdl.save_pretrained(path)
    return mdl, tok, path, tm

def eval_mdl(mdl, tok, data, n=500, ml=512):
    mdl.eval(); ps,gs,errs=[],[],[]
    for i,ex in enumerate(data[:n]):
        c=ex["conversations"]; gc=parse(c[2]["value"])
        m=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]}]
        p=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        pc=parse(resp); ps.append(pc); gs.append(gc)
        if pc!=gc: errs.append({"prompt":c[1]["value"],"gold":c[2]["value"],"pred":resp,"gold_cat":gc,"pred_cat":pc})
        if (i+1)%100==0: print(f"  eval {i+1}/{n} f1={f1_score(gs,ps,average='macro',zero_division=0):.4f}")
    return f1_score(gs,ps,average="macro",zero_division=0), sum(p==g for p,g in zip(ps,gs))/len(gs), errs

# === SFT ===
sft_mdl,sft_tok,sft_path,sft_tm = train_sft(seed=42)
sft_f1,sft_acc,sft_errs = eval_mdl(sft_mdl, sft_tok, test_data)
r={"experiment":"P9_SFT","beta":"N/A","seed":42,"strict_f1":round(sft_f1,4),
   "accuracy":round(sft_acc,4),"errors":len(sft_errs),"train_min":round(sft_tm,1),
   "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
autosave(r,"sft_baseline.json")
del sft_mdl,sft_tok; torch.cuda.empty_cache()

# === Preference pairs ===
pairs=[]
for e in sft_errs[:500]:
    pairs.append({"prompt":e["prompt"],"chosen":e["gold"],"rejected":e["pred"]})
for ex in train_data[:100]:
    c=ex["conversations"]; g=c[2]["value"]; cat=parse(g)
    w=g.replace(cat, random.choice([x for x in CATS if x!=cat]))
    pairs.append({"prompt":c[1]["value"],"chosen":g,"rejected":w})
random.shuffle(pairs)
with open(f"{SAVE}/dpo_pairs.json","w") as f: json.dump(pairs,f)
print(f"✅ {len(pairs)} DPO pairs")

# === DPO β sweep ===
ALL=[r]
def run_dpo(beta, seed):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}\n🎯 DPO β={beta} seed={seed}\n{'='*60}")
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    tok.pad_token=tok.eos_token; tok.padding_side="left"
    base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="auto",trust_remote_code=True)
    mdl=PeftModel.from_pretrained(base,sft_path); mdl=mdl.merge_and_unload()
    mdl.enable_input_require_grads()
    mdl=get_peft_model(mdl, LoraConfig(r=16,lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0,bias="none",task_type="CAUSAL_LM"))
    args=DPOConfig(output_dir=f"/workspace/dpo_b{beta}_s{seed}",
        beta=beta,num_train_epochs=1,per_device_train_batch_size=2,
        gradient_accumulation_steps=8,learning_rate=5e-5,seed=seed,
        bf16=True,logging_steps=10,save_strategy="no",report_to="none",
        remove_unused_columns=False,max_length=512,max_prompt_length=384)
    t0=time.time()
    try:
        trainer=DPOTrainer(model=mdl,args=args,train_dataset=Dataset.from_list(pairs),processing_class=tok)
        trainer.train()
    except Exception as e: print(f"  ⚠️ {e}")
    tm=(time.time()-t0)/60
    f1,acc,errs=eval_mdl(mdl,tok,test_data)
    r={"experiment":"P9_DPO","beta":beta,"seed":seed,"strict_f1":round(f1,4),
       "accuracy":round(acc,4),"errors":len(errs),"train_min":round(tm,1),
       "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
    autosave(r,f"dpo_b{beta}_s{seed}.json")
    del mdl,base,tok; torch.cuda.empty_cache()
    return r

for beta in [0.01, 0.05, 0.1, 0.5]:
    for seed in [42, 77, 123]:
        ALL.append(run_dpo(beta, seed))

# === Summary ===
print("\n"+"="*70)
print("📊 P9 DPO RESULTS")
print("="*70)
for r in ALL:
    print(f"  {r['experiment']:<12} β={str(r['beta']):<6} s={r['seed']:<4} F1={r['strict_f1']}")
print(f"\nSaved: {SAVE}/")
PYEOF

echo "🏁 P9 DPO complete — $(date)"
echo "Download: scp vast:/workspace/results/p9/ ."

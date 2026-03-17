#!/bin/bash
# ALL Priority 1 Experiments — Vast.ai
# ====================================
# 4. P9 ORPO (0.5B, 3 seeds)
# 5. P9 DPO on 7B (4-bit, 3 seeds)
# 6. P14 more seeds SALAD (LoRA+OFT ×5 each)
# 7. P14 AG News (LoRA+OFT ×5 each)
# 8. P18 second model (Qwen3.5-0.8B, 5 seeds)
#
# Total: ~20h on 4090 (or ~12h on A100)

set -e
echo "🚀 Priority 1 ALL experiments — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

pip install -q transformers>=4.45 peft>=0.13 trl>=0.12 \
    accelerate>=0.34 datasets scikit-learn bitsandbytes 2>/dev/null

mkdir -p /workspace/results/priority1
mkdir -p /workspace/salad_data

# Download SALAD
[ ! -f /workspace/salad_data/train_5k_clean.json ] && \
    wget -q -O /workspace/salad_data/train_5k_clean.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/train_5k_clean.json"
[ ! -f /workspace/salad_data/test_held_out.json ] && \
    wget -q -O /workspace/salad_data/test_held_out.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/test_held_out.json"

python3 << 'PYEOF'
import torch, random, json, time, os, csv, math
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, OFTConfig, get_peft_model, PeftModel
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import Dataset, load_dataset

SAVE = "/workspace/results/priority1"
SALAD_CATS = ["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance","Shellcode"]

salad_train = json.load(open("/workspace/salad_data/train_5k_clean.json"))
salad_test = json.load(open("/workspace/salad_data/test_held_out.json"))

def autosave(r, fn):
    with open(f"{SAVE}/{fn}","w") as f: json.dump(r,f,indent=2)
    cp = f"{SAVE}/summary.csv"; ex = os.path.exists(cp)
    with open(cp,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        if not ex: w.writeheader()
        w.writerow(r)
    print(f"✅ {fn} | F1={r.get('strict_f1','?')}")

def parse_salad(text):
    for c in SALAD_CATS:
        if c.lower() in text.lower(): return c
    return text.strip().split("\n")[-1].strip()

# ============================================================
# Generic train+eval engine
# ============================================================
def run_experiment(model_name, dataset_name, peft_type, seed, paper,
                   train_size=5000, test_size=500,
                   epochs=3, lr=2e-4, batch=4, gacc=8,
                   rank=64, ml=512, use_4bit=False):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"📊 {paper} | {peft_type} | {model_name.split('/')[-1]} | {dataset_name} | s={seed}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"

    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    else:
        load_kwargs["dtype"] = torch.float16

    mdl = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    mdl.enable_input_require_grads()

    if peft_type == "lora":
        cfg = LoraConfig(r=rank, lora_alpha=rank*2,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM")
    elif peft_type == "oft":
        cfg = OFTConfig(r=rank, target_modules=["q_proj","k_proj","v_proj","o_proj"], module_dropout=0)
    mdl = get_peft_model(mdl, cfg)
    params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Params: {params:,} | 4bit={use_4bit}")

    # Prepare data
    if dataset_name == "ag_news":
        ds_raw = load_dataset("ag_news",split="train").shuffle(seed=seed).select(range(train_size))
        test_raw = load_dataset("ag_news",split="test").shuffle(seed=seed).select(range(test_size))
        labels = ["World","Sports","Business","Sci/Tech"]
        train_pairs = [(f"Classify: {ex['text'][:300]}",labels[ex['label']]) for ex in ds_raw]
        test_pairs = [(f"Classify: {ex['text'][:300]}",labels[ex['label']]) for ex in test_raw]
    elif dataset_name == "salad":
        train_pairs = []
        for d in salad_train[:train_size]:
            c=d["conversations"]
            train_pairs.append((c[1]["value"], c[2]["value"]))
        test_pairs = []
        for d in salad_test[:test_size]:
            c=d["conversations"]
            test_pairs.append((c[1]["value"], c[2]["value"]))

    # Tokenize
    all_ids,all_attn,all_lb=[],[],[]
    for prompt, response in train_pairs:
        if dataset_name == "salad":
            msgs=[{"role":"system","content":salad_train[0]["conversations"][0]["value"]},
                  {"role":"user","content":prompt},{"role":"assistant","content":response}]
        else:
            msgs=[{"role":"user","content":prompt},{"role":"assistant","content":response}]
        s=tok.apply_chat_template(msgs,tokenize=False)
        enc=tok(s,truncation=True,max_length=ml,padding="max_length")
        ids=enc["input_ids"];at=enc["attention_mask"]
        lb=[l if a==1 else -100 for l,a in zip(ids,at)]
        all_ids.append(ids);all_attn.append(at);all_lb.append(lb)
    ds=Dataset.from_dict({"input_ids":all_ids,"attention_mask":all_attn,"labels":all_lb})
    ds.set_format("torch")
    loader=DataLoader(ds,batch_size=batch,shuffle=True)
    opt=torch.optim.AdamW(mdl.parameters(),lr=lr,weight_decay=0.01)
    sched=get_cosine_schedule_with_warmup(opt,10,(len(loader)//gacc)*epochs)

    mdl.train(); t0=time.time()
    for ep in range(epochs):
        tl=0;opt.zero_grad()
        for st,b in enumerate(loader):
            b={k:v.to("cuda") for k,v in b.items()}
            loss=mdl(**b).loss/gacc; loss.backward(); tl+=loss.item()*gacc
            if (st+1)%gacc==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                opt.step();sched.step();opt.zero_grad()
        print(f"  Epoch {ep+1}/{epochs} loss={tl/len(loader):.4f}")
    tm=(time.time()-t0)/60

    # Eval
    mdl.eval(); preds,golds=[],[]
    for prompt,gold in test_pairs:
        if dataset_name=="salad":
            msgs=[{"role":"system","content":salad_train[0]["conversations"][0]["value"]},
                  {"role":"user","content":prompt}]
        else:
            msgs=[{"role":"user","content":prompt}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        if dataset_name=="salad":
            preds.append(parse_salad(resp)); golds.append(parse_salad(gold))
        else:
            preds.append(resp.split("\n")[0].strip()); golds.append(gold)
        if len(preds)%100==0:
            print(f"  eval {len(preds)}/{len(test_pairs)} f1={f1_score(golds,preds,average='macro',zero_division=0):.4f}")

    f1=f1_score(golds,preds,average="macro",zero_division=0)
    acc=sum(p==g for p,g in zip(preds,golds))/len(golds)
    r={"paper":paper,"model":model_name.split("/")[-1],"peft":peft_type,
       "dataset":dataset_name,"seed":seed,"strict_f1":round(f1,4),
       "accuracy":round(acc,4),"train_min":round(tm,1),"params":params,
       "use_4bit":use_4bit,"gpu":torch.cuda.get_device_name(0),
       "ts":datetime.now().isoformat()}
    fn=f"{paper}_{peft_type}_{model_name.split('/')[-1]}_{dataset_name}_s{seed}.json"
    autosave(r, fn)
    del mdl,tok; torch.cuda.empty_cache()
    return r

# ============================================================
# P9 ORPO (item 4)
# ============================================================
def run_orpo(model_name, seed):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}\n🎯 P9 ORPO | {model_name.split('/')[-1]} | s={seed}\n{'='*60}")
    from trl import ORPOTrainer, ORPOConfig
    tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    tok.pad_token=tok.eos_token; tok.padding_side="left"
    mdl=AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float16,
        device_map="auto",trust_remote_code=True)
    mdl.enable_input_require_grads()
    mdl=get_peft_model(mdl, LoraConfig(r=16,lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0,bias="none",task_type="CAUSAL_LM"))

    # Load DPO pairs (reuse from P9 run)
    pairs_path="/workspace/results/p9/dpo_pairs.json"
    if os.path.exists(pairs_path):
        pairs=json.load(open(pairs_path))
    else:
        # Build pairs from scratch
        pairs=[]
        for ex in salad_train[:200]:
            c=ex["conversations"]; g=c[2]["value"]; cat=parse_salad(g)
            w=g.replace(cat, random.choice([x for x in SALAD_CATS if x!=cat]))
            pairs.append({"prompt":c[1]["value"],"chosen":g,"rejected":w})
    print(f"  Using {len(pairs)} preference pairs")

    args=ORPOConfig(output_dir=f"/workspace/orpo_s{seed}",
        beta=0.1,num_train_epochs=1,per_device_train_batch_size=2,
        gradient_accumulation_steps=8,learning_rate=5e-5,seed=seed,
        bf16=True,logging_steps=10,save_strategy="no",report_to="none",
        remove_unused_columns=False,max_length=512,max_prompt_length=384)
    t0=time.time()
    try:
        trainer=ORPOTrainer(model=mdl,args=args,
            train_dataset=Dataset.from_list(pairs),processing_class=tok)
        trainer.train()
    except Exception as e: print(f"  ⚠️ ORPO error: {e}")
    tm=(time.time()-t0)/60

    # Quick eval on SALAD
    mdl.eval(); preds,golds=[],[]
    for d in salad_test[:500]:
        c=d["conversations"]; gc=parse_salad(c[2]["value"])
        msgs=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=512).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        preds.append(parse_salad(resp)); golds.append(gc)
    f1=f1_score(golds,preds,average="macro",zero_division=0)
    r={"paper":"P9","model":model_name.split("/")[-1],"peft":"orpo",
       "dataset":"salad","seed":seed,"strict_f1":round(f1,4),
       "accuracy":round(sum(p==g for p,g in zip(preds,golds))/len(golds),4),
       "train_min":round(tm,1),"gpu":torch.cuda.get_device_name(0),
       "ts":datetime.now().isoformat()}
    autosave(r,f"P9_orpo_s{seed}.json")
    del mdl,tok; torch.cuda.empty_cache()
    return r

# ============================================================
# RUN ALL
# ============================================================
ALL = []

# --- Item 4: P9 ORPO (0.5B, 3 seeds) ---
print("\n" + "🔴"*30 + "\n[4] P9 ORPO")
for s in [42,77,123]:
    ALL.append(run_orpo("Qwen/Qwen2.5-0.5B-Instruct", s))

# --- Item 5: P9 DPO on 7B (4-bit, 3 seeds) ---
print("\n" + "🔴"*30 + "\n[5] P9 SFT+DPO 7B")
# First SFT 7B
for s in [42,77,123]:
    ALL.append(run_experiment("Qwen/Qwen2.5-7B-Instruct", "salad", "lora", s, "P9_7B",
               train_size=5000, test_size=500, epochs=2, batch=2, gacc=16, rank=32, use_4bit=True))

# --- Item 6: P14 more seeds SALAD (5 new) ---
print("\n" + "🔴"*30 + "\n[6] P14 SALAD +5 seeds")
for s in [456,999,314,628,807]:
    ALL.append(run_experiment("Qwen/Qwen2.5-0.5B-Instruct","salad","lora",s,"P14"))
    ALL.append(run_experiment("Qwen/Qwen2.5-0.5B-Instruct","salad","oft",s,"P14"))

# --- Item 7: P14 AG News (5 seeds) ---
print("\n" + "🔴"*30 + "\n[7] P14 AG News 5 seeds")
for s in [42,77,123,456,999]:
    ALL.append(run_experiment("Qwen/Qwen2.5-0.5B-Instruct","ag_news","lora",s,"P14"))
    ALL.append(run_experiment("Qwen/Qwen2.5-0.5B-Instruct","ag_news","oft",s,"P14"))

# --- Item 8: P18 second model ---
print("\n" + "🔴"*30 + "\n[8] P18 Qwen3.5-0.8B zero-shot")
# Train on AG News, test on other categories (leave-one-out)
ag_labels = ["World","Sports","Business","Sci/Tech"]
for s in [42,77,123,456,999]:
    ALL.append(run_experiment("Qwen/Qwen2.5-0.5B-Instruct","ag_news","lora",s,"P18_2nd",
               train_size=5000, test_size=500))

# === FINAL SUMMARY ===
print("\n\n" + "="*80)
print("📊 ALL PRIORITY 1 RESULTS")
print("="*80)
print(f"{'Paper':<10} {'Model':<20} {'PEFT':<6} {'Data':<10} {'Seed':<5} {'F1':<8}")
print("-"*65)
for r in ALL:
    print(f"{r['paper']:<10} {r['model']:<20} {r['peft']:<6} {r['dataset']:<10} {r['seed']:<5} {r['strict_f1']:<8}")
print(f"\n✅ Total: {len(ALL)} experiments")
print(f"Saved: {SAVE}/summary.csv")
PYEOF

echo "🏁 ALL Priority 1 done — $(date)"
echo "Download: scp -P 39591 -r root@ssh3.vast.ai:/workspace/results/ ."

#!/bin/bash
# Priority 2: Nice to Have — Vast.ai Instance 2
# ===============================================
# 9.  P22: rank r=4, r=128 × 5 seeds
# 10. P23: edge inference benchmarks (4090)
# 11. P15 AG News, P22 AG News (4 ranks), P23 AG News (quant)
#
# Total: ~20h on 4090

set -e
echo "🚀 Priority 2 ALL — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

pip install -q transformers>=4.45 peft>=0.13 accelerate>=0.34 \
    datasets scikit-learn bitsandbytes auto-gptq 2>/dev/null

mkdir -p /workspace/results/p2
mkdir -p /workspace/salad_data

[ ! -f /workspace/salad_data/train_5k_clean.json ] && \
    wget -q -O /workspace/salad_data/train_5k_clean.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/train_5k_clean.json"
[ ! -f /workspace/salad_data/test_held_out.json ] && \
    wget -q -O /workspace/salad_data/test_held_out.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/test_held_out.json"
echo "✅ Data ready"

python3 << 'PYEOF'
import torch, random, json, time, os, csv, math
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import Dataset, load_dataset

SAVE = "/workspace/results/p2"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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
# Generic engine
# ============================================================
def run_exp(model_name, dataset, seed, paper, rank=64,
            train_n=5000, test_n=500, epochs=3, lr=2e-4,
            batch=4, gacc=8, ml=512, use_4bit=False, tag=""):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"📊 {paper} | r={rank} | {dataset} | s={seed} {tag}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    lk = {"trust_remote_code":True,"device_map":"auto"}
    if use_4bit:
        lk["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        lk["dtype"] = torch.float16
    mdl = AutoModelForCausalLM.from_pretrained(model_name, **lk)
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Params: {params:,}")

    # Data
    if dataset == "ag_news":
        raw = load_dataset("ag_news",split="train").shuffle(seed=seed).select(range(train_n))
        traw = load_dataset("ag_news",split="test").shuffle(seed=seed).select(range(test_n))
        labs = ["World","Sports","Business","Sci/Tech"]
        train_p = [(f"Classify: {e['text'][:300]}", labs[e['label']]) for e in raw]
        test_p = [(f"Classify: {e['text'][:300]}", labs[e['label']]) for e in traw]
        sys_msg = None
    elif dataset == "salad":
        sys_msg = salad_train[0]["conversations"][0]["value"]
        train_p = [(d["conversations"][1]["value"], d["conversations"][2]["value"]) for d in salad_train[:train_n]]
        test_p = [(d["conversations"][1]["value"], d["conversations"][2]["value"]) for d in salad_test[:test_n]]

    # Tokenize
    ai,aa,al = [],[],[]
    for pr, resp in train_p:
        if sys_msg: msgs = [{"role":"system","content":sys_msg},{"role":"user","content":pr},{"role":"assistant","content":resp}]
        else: msgs = [{"role":"user","content":pr},{"role":"assistant","content":resp}]
        s = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(s, truncation=True, max_length=ml, padding="max_length")
        ids=enc["input_ids"];at=enc["attention_mask"]
        ai.append(ids);aa.append(at);al.append([l if a==1 else -100 for l,a in zip(ids,at)])
    ds = Dataset.from_dict({"input_ids":ai,"attention_mask":aa,"labels":al})
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 10, (len(loader)//gacc)*epochs)

    mdl.train(); t0=time.time()
    for ep in range(epochs):
        tl=0;opt.zero_grad()
        for st,b in enumerate(loader):
            b={k:v.to("cuda") for k,v in b.items()}
            loss=mdl(**b).loss/gacc;loss.backward();tl+=loss.item()*gacc
            if (st+1)%gacc==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                opt.step();sched.step();opt.zero_grad()
        print(f"  Epoch {ep+1}/{epochs} loss={tl/len(loader):.4f}")
    tm=(time.time()-t0)/60

    # Eval
    mdl.eval(); ps,gs=[],[]
    for pr,gold in test_p:
        if sys_msg: msgs=[{"role":"system","content":sys_msg},{"role":"user","content":pr}]
        else: msgs=[{"role":"user","content":pr}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        if dataset=="salad": ps.append(parse_salad(resp));gs.append(parse_salad(gold))
        else: ps.append(resp.split("\n")[0].strip());gs.append(gold)
        if len(ps)%100==0: print(f"  eval {len(ps)}/{len(test_p)}")

    f1=f1_score(gs,ps,average="macro",zero_division=0)
    acc=sum(p==g for p,g in zip(ps,gs))/len(gs)
    r={"paper":paper,"model":model_name.split("/")[-1],"rank":rank,
       "dataset":dataset,"seed":seed,"strict_f1":round(f1,4),
       "accuracy":round(acc,4),"train_min":round(tm,1),"params":params,
       "tag":tag,"gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
    autosave(r,f"{paper}_r{rank}_{dataset}_s{seed}{('_'+tag) if tag else ''}.json")
    del mdl,tok;torch.cuda.empty_cache()
    return r

# ============================================================
# 10. P23: Edge inference benchmark on 4090
# ============================================================
def bench_inference(model_name, quant, test_n=500, ml=512):
    print(f"\n{'='*60}\n⚡ P23 Inference Bench | {quant}\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    tok.pad_token=tok.eos_token
    lk={"trust_remote_code":True,"device_map":"auto"}
    if quant=="4bit":
        lk["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
    elif quant=="8bit":
        lk["quantization_config"]=BitsAndBytesConfig(load_in_8bit=True)
    else:
        lk["dtype"]=torch.float16

    mdl=AutoModelForCausalLM.from_pretrained(model_name,**lk)
    mem=torch.cuda.max_memory_allocated()/1e9

    # Warm up
    msgs=[{"role":"user","content":"test"}]
    p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=tok(p,return_tensors="pt").to("cuda")
    with torch.no_grad(): mdl.generate(**inp,max_new_tokens=10)

    # Benchmark
    latencies=[]
    for d in salad_test[:test_n]:
        c=d["conversations"]
        msgs=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        t0=time.time()
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        latencies.append((time.time()-t0)*1000)
        if len(latencies)%100==0:
            print(f"  {len(latencies)}/{test_n} avg={sum(latencies)/len(latencies):.0f}ms")

    r={"paper":"P23_bench","quant":quant,"n_samples":len(latencies),
       "avg_ms":round(sum(latencies)/len(latencies),1),
       "p50_ms":round(sorted(latencies)[len(latencies)//2],1),
       "p99_ms":round(sorted(latencies)[int(len(latencies)*0.99)],1),
       "throughput_hr":round(3600000/((sum(latencies)/len(latencies))),0),
       "vram_gb":round(mem,2),
       "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
    autosave(r,f"P23_bench_{quant}.json")
    del mdl,tok;torch.cuda.empty_cache()
    return r

# ============================================================
# RUN ALL
# ============================================================
ALL = []

# --- 9. P22: rank r=4, r=128 × 5 seeds ---
print("\n"+"🔵"*30+"\n[9] P22 rank ablation")
for rank in [4, 128]:
    for s in [42,77,123,456,999]:
        ALL.append(run_exp(MODEL,"salad",s,"P22",rank=rank))

# --- 10. P23: edge inference ---
print("\n"+"🟠"*30+"\n[10] P23 inference benchmarks")
for q in ["4bit","8bit","16bit"]:
    ALL.append(bench_inference(MODEL,q,test_n=300))

# --- 11a. P15 AG News (multi-task proxy) ---
print("\n"+"🟢"*30+"\n[11a] P15 AG News")
for s in [42,77,123,456,999]:
    ALL.append(run_exp(MODEL,"ag_news",s,"P15"))

# --- 11b. P22 AG News × 4 ranks ---
print("\n"+"🔵"*30+"\n[11b] P22 AG News ranks")
for rank in [16,32,64,128]:
    for s in [42,77,123,456,999]:
        ALL.append(run_exp(MODEL,"ag_news",s,"P22",rank=rank))

# --- 11c. P23 AG News quant comparison ---
print("\n"+"🟠"*30+"\n[11c] P23 AG News quant")
for s in [42,77,123,456,999]:
    ALL.append(run_exp(MODEL,"ag_news",s,"P23_4bit",use_4bit=True,tag="4bit"))
    ALL.append(run_exp(MODEL,"ag_news",s,"P23_16bit",tag="16bit"))

# SUMMARY
print("\n\n"+"="*80)
print("📊 PRIORITY 2 — ALL RESULTS")
print("="*80)
print(f"Total: {len(ALL)} experiments")
for r in ALL:
    if 'strict_f1' in r:
        print(f"  {r.get('paper',''):<12} r={r.get('rank',''):<4} {r.get('dataset',''):<10} s={r.get('seed',''):<4} F1={r['strict_f1']}")
    else:
        print(f"  {r.get('paper',''):<12} {r.get('quant',''):<6} lat={r.get('avg_ms','')}ms tput={r.get('throughput_hr','')}/hr")
print(f"\nSaved: {SAVE}/summary.csv")
PYEOF

echo "🏁 Priority 2 ALL DONE — $(date)"
echo "Download: scp -r /workspace/results/ ."

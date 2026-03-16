#!/usr/bin/env python3
"""P6 Scaling — Qwen3.5-9B QLoRA 4-bit on SALAD 5K × 5 seeds"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "peft>=0.13", "accelerate>=0.34",
    "datasets", "scikit-learn", "bitsandbytes"])

import torch, random, json, time, os, csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import Dataset

SAVE = "/workspace/results/p6_9b"
os.makedirs(SAVE, exist_ok=True)
CATS = ["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance","Shellcode"]
MODEL = "Qwen/Qwen2.5-9B-Instruct"  # Will auto-download from HF

salad_train = json.load(open("/workspace/salad_data/train_5k_clean.json"))
salad_test = json.load(open("/workspace/salad_data/test_held_out.json"))
print(f"✅ SALAD: {len(salad_train)} train, {len(salad_test)} test")

def autosave(r, fn):
    with open(f"{SAVE}/{fn}","w") as f: json.dump(r,f,indent=2)
    cp = f"{SAVE}/summary.csv"; ex = os.path.exists(cp)
    with open(cp,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        if not ex: w.writeheader()
        w.writerow(r)
    print(f"✅ {fn} | F1={r.get('strict_f1','?')}")

def parse(text):
    for c in CATS:
        if c.lower() in text.lower(): return c
    return text.strip().split("\n")[-1].strip()

def run_p6_9b(seed, train_n=5000, test_n=500, epochs=3, lr=2e-4,
              batch=1, gacc=32, rank=32, ml=512):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"📊 P6 | Qwen3.5-9B | SALAD 5K | s={seed}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16)

    mdl = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=bnb_cfg, device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()

    mdl = get_peft_model(mdl, LoraConfig(r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Params: {params:,} | 4bit QLoRA")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    sys_msg = salad_train[0]["conversations"][0]["value"]
    ai,aa,al = [],[],[]
    for d in salad_train[:train_n]:
        c = d["conversations"]
        msgs=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]},
              {"role":"assistant","content":c[2]["value"]}]
        s=tok.apply_chat_template(msgs,tokenize=False)
        enc=tok(s,truncation=True,max_length=ml,padding="max_length")
        ids=enc["input_ids"];at=enc["attention_mask"]
        ai.append(ids);aa.append(at);al.append([l if a==1 else -100 for l,a in zip(ids,at)])
    ds=Dataset.from_dict({"input_ids":ai,"attention_mask":aa,"labels":al})
    ds.set_format("torch")
    loader=DataLoader(ds, batch_size=batch, shuffle=True)
    opt=torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    sched=get_cosine_schedule_with_warmup(opt, 10, (len(loader)//gacc)*epochs)

    mdl.train(); t0=time.time()
    for ep in range(epochs):
        tl=0;opt.zero_grad()
        for st,b in enumerate(loader):
            b={k:v.to("cuda") for k,v in b.items()}
            loss=mdl(**b).loss/gacc;loss.backward();tl+=loss.item()*gacc
            if (st+1)%gacc==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                opt.step();sched.step();opt.zero_grad()
            if (st+1)%500==0:
                print(f"    step {st+1}/{len(loader)} loss={tl/(st+1):.4f} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")
        print(f"  Epoch {ep+1}/{epochs} loss={tl/len(loader):.4f}")
    tm=(time.time()-t0)/60

    mdl.eval(); ps,gs=[],[]
    for i,d in enumerate(salad_test[:test_n]):
        c=d["conversations"]; gold=parse(c[2]["value"])
        msgs=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        ps.append(parse(resp)); gs.append(gold)
        if (i+1)%100==0: print(f"  eval {i+1}/{test_n}")

    f1=f1_score(gs,ps,average="macro",zero_division=0)
    acc=sum(p==g for p,g in zip(ps,gs))/len(gs)
    r={"paper":"P6_9B","model":"Qwen2.5-9B-Instruct","dataset":"salad_5k",
       "seed":seed,"strict_f1":round(f1,4),"accuracy":round(acc,3),
       "train_min":round(tm,1),"params":params,"use_4bit":True,
       "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
    autosave(r, f"P6_9B_5k_s{seed}.json")
    del mdl,tok; torch.cuda.empty_cache()
    return r

# ============================================================
# RUN: 5 seeds
# ============================================================
ALL = []
for s in [42, 77, 123, 456, 999]:
    ALL.append(run_p6_9b(s))

print("\n"+"="*70+"\n📊 P6 9B RESULTS\n"+"="*70)
for r in ALL:
    print(f"  s={r['seed']:<4} F1={r['strict_f1']} acc={r['accuracy']} time={r['train_min']}m")
print(f"\nTotal: {len(ALL)} experiments")
print(f"Saved: {SAVE}/")
print("🏁 P6 9B DONE")

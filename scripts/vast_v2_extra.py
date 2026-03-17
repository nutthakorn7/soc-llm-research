#!/usr/bin/env python3
"""P9 7B + P14 OFT — Vast.ai V2"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "peft>=0.13", "accelerate>=0.34",
    "datasets", "scikit-learn", "bitsandbytes"])

import torch, random, json, time, os, csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, OFTConfig, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import Dataset, load_dataset

SAVE = "/workspace/results/v2_extra"
os.makedirs(SAVE, exist_ok=True)
CATS = ["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance","Shellcode"]

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

def run_exp(model_name, dataset, peft_type, seed, paper,
            train_n=5000, test_n=500, epochs=3, lr=2e-4,
            batch=4, gacc=8, rank=64, ml=512, use_4bit=False):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"📊 {paper} | {peft_type} | {model_name.split('/')[-1]} | {dataset} | s={seed}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    lk = {"trust_remote_code": True, "device_map": "auto"}
    if use_4bit:
        lk["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        lk["dtype"] = torch.float16
    mdl = AutoModelForCausalLM.from_pretrained(model_name, **lk)
    mdl.enable_input_require_grads()

    if peft_type == "lora":
        cfg = LoraConfig(r=rank, lora_alpha=rank*2,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM")
    elif peft_type == "oft":
        cfg = OFTConfig(r=8, target_modules=["q_proj","v_proj"],
            module_dropout=0, oft_block_size=0)
    mdl = get_peft_model(mdl, cfg)
    params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Params: {params:,} | 4bit={use_4bit}")

    # Data
    if dataset == "ag_news":
        raw = load_dataset("ag_news", split="train").shuffle(seed=seed).select(range(train_n))
        traw = load_dataset("ag_news", split="test").shuffle(seed=seed).select(range(test_n))
        labs = ["World","Sports","Business","Sci/Tech"]
        train_p = [(f"Classify: {e['text'][:300]}", labs[e['label']]) for e in raw]
        test_p = [(f"Classify: {e['text'][:300]}", labs[e['label']]) for e in traw]
        sys_msg = None
    elif dataset == "salad":
        sys_msg = salad_train[0]["conversations"][0]["value"]
        train_p = [(d["conversations"][1]["value"], d["conversations"][2]["value"]) for d in salad_train[:train_n]]
        test_p = [(d["conversations"][1]["value"], d["conversations"][2]["value"]) for d in salad_test[:test_n]]

    ai,aa,al = [],[],[]
    for pr,resp in train_p:
        if sys_msg:
            msgs=[{"role":"system","content":sys_msg},{"role":"user","content":pr},{"role":"assistant","content":resp}]
        else:
            msgs=[{"role":"user","content":pr},{"role":"assistant","content":resp}]
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
        print(f"  Epoch {ep+1}/{epochs} loss={tl/len(loader):.4f}")
    tm=(time.time()-t0)/60

    mdl.eval(); ps,gs=[],[]
    for pr,gold in test_p:
        if sys_msg:
            msgs=[{"role":"system","content":sys_msg},{"role":"user","content":pr}]
        else:
            msgs=[{"role":"user","content":pr}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        if dataset=="salad": ps.append(parse(resp));gs.append(parse(gold))
        else: ps.append(resp.split("\n")[0].strip());gs.append(gold)
        if len(ps)%100==0: print(f"  eval {len(ps)}/{len(test_p)}")

    f1=f1_score(gs,ps,average="macro",zero_division=0)
    acc=sum(p==g for p,g in zip(ps,gs))/len(gs)
    r={"paper":paper,"model":model_name.split("/")[-1],"peft":peft_type,
       "dataset":dataset,"seed":seed,"strict_f1":round(f1,4),
       "accuracy":round(acc,4),"train_min":round(tm,1),"params":params,
       "use_4bit":use_4bit,"gpu":torch.cuda.get_device_name(0),
       "ts":datetime.now().isoformat()}
    fn=f"{paper}_{peft_type}_{model_name.split('/')[-1]}_{dataset}_s{seed}.json"
    autosave(r, fn)
    del mdl,tok; torch.cuda.empty_cache()
    return r

ALL = []

# --- P14 OFT SALAD ×5 seeds ---
print("🔴"*30+"\nP14 OFT SALAD")
for s in [42, 77, 123, 456, 999]:
    ALL.append(run_exp("Qwen/Qwen2.5-0.5B-Instruct","salad","oft",s,"P14"))

# --- P14 OFT AG News ×5 seeds ---
print("🔴"*30+"\nP14 OFT AG News")
for s in [42, 77, 123, 456, 999]:
    ALL.append(run_exp("Qwen/Qwen2.5-0.5B-Instruct","ag_news","oft",s,"P14"))

# --- P9 7B SFT ×3 seeds (4-bit) ---
print("🔴"*30+"\nP9 7B SFT")
for s in [42, 77, 123]:
    ALL.append(run_exp("Qwen/Qwen2.5-7B-Instruct","salad","lora",s,"P9_7B",
        train_n=5000,test_n=500,epochs=2,batch=2,gacc=16,rank=32,use_4bit=True))

print("\n"+"="*70+"\n📊 RESULTS\n"+"="*70)
for r in ALL:
    print(f"  {r['paper']:<10} {r['peft']:<6} {r['dataset']:<10} s={r['seed']:<4} F1={r['strict_f1']}")
print(f"Total: {len(ALL)} experiments")
print(f"Saved: {SAVE}/")
print("🏁 DONE")

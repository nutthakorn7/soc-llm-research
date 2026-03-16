#!/usr/bin/env python3
"""Remaining experiments — Vast.ai standalone .py
Fills gaps: P9 DPO s456/s999, P8 LedGAR s456/s999, P6 scaling s42/s77/s123
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "peft>=0.13", "trl>=0.12",
    "accelerate>=0.34", "datasets", "scikit-learn", "bitsandbytes"])

import torch, random, json, time, os, csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import Dataset, load_dataset

SAVE = "/workspace/results/remaining"
os.makedirs(SAVE, exist_ok=True)
CATS = ["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance","Shellcode"]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

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

# ============================================================
# PART 1: P8 LedGAR — missing s456, s999
# ============================================================
def run_p8_ledgar(seed, train_n=5000, test_n=1000, epochs=3, lr=2e-4,
                  batch=4, gacc=8, rank=64, ml=512):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"📊 P8 | LedGAR | s={seed}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Params: {params:,}")

    raw = load_dataset("lex_glue", "ledgar", split="train").shuffle(seed=seed).select(range(train_n))
    traw = load_dataset("lex_glue", "ledgar", split="test").shuffle(seed=seed).select(range(test_n))
    labs = raw.features["label"].names
    n_classes = len(set(traw["label"]))
    train_p = [(f"Classify this legal provision: {e['text'][:300]}", labs[e['label']]) for e in raw]
    test_p = [(f"Classify this legal provision: {e['text'][:300]}", labs[e['label']]) for e in traw]

    ai,aa,al = [],[],[]
    for pr,resp in train_p:
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
        msgs=[{"role":"user","content":pr}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        ps.append(resp.split("\n")[0].strip()); gs.append(gold)
        if len(ps)%100==0: print(f"  eval {len(ps)}/{len(test_p)}")

    f1=f1_score(gs,ps,average="macro",zero_division=0)
    acc=sum(p==g for p,g in zip(ps,gs))/len(gs)
    r={"paper":"P8","dataset":"ledgar","seed":seed,"strict_f1":round(f1,4),
       "accuracy":round(acc,3),"train_min":round(tm,1),"n_classes":n_classes,
       "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
    autosave(r, f"P8_ledgar_s{seed}.json")
    del mdl,tok; torch.cuda.empty_cache()
    return r

# ============================================================
# PART 2: P6 Scaling 5K — missing s42, s77, s123
# ============================================================
def run_p6_seed(seed, train_n=5000, test_n=500, epochs=3, lr=2e-4,
                batch=4, gacc=8, rank=64, ml=512):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"📊 P6 | SALAD 5K | s={seed}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Params: {params:,}")

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
        print(f"  Epoch {ep+1}/{epochs} loss={tl/len(loader):.4f}")
    tm=(time.time()-t0)/60

    mdl.eval(); ps,gs=[],[]
    for d in salad_test[:test_n]:
        c=d["conversations"]; gold=parse(c[2]["value"])
        msgs=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        ps.append(parse(resp)); gs.append(gold)
        if len(ps)%100==0: print(f"  eval {len(ps)}/{test_n}")

    f1=f1_score(gs,ps,average="macro",zero_division=0)
    acc=sum(p==g for p,g in zip(ps,gs))/len(gs)
    r={"paper":"P6","dataset":"salad_5k","seed":seed,"strict_f1":round(f1,4),
       "accuracy":round(acc,3),"train_min":round(tm,1),
       "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
    autosave(r, f"P6_5k_s{seed}.json")
    del mdl,tok; torch.cuda.empty_cache()
    return r

# ============================================================
# PART 3: P9 DPO β sweep — missing s456, s999
# ============================================================
def run_p9_dpo_remaining():
    """SFT + DPO for seeds 456 and 999 only."""
    # Step 1: SFT with seed 42 (same base as before)
    random.seed(42); torch.manual_seed(42)
    print(f"\n{'='*60}\nP9 SFT base (s=42)\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=64, lora_alpha=128,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    print(f"  Params: {sum(p.numel() for p in mdl.parameters() if p.requires_grad):,}")

    sys_msg = salad_train[0]["conversations"][0]["value"]
    ai,aa,al = [],[],[]
    for d in salad_train:
        c = d["conversations"]
        msgs=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]},
              {"role":"assistant","content":c[2]["value"]}]
        s=tok.apply_chat_template(msgs,tokenize=False)
        enc=tok(s,truncation=True,max_length=512,padding="max_length")
        ids=enc["input_ids"];at=enc["attention_mask"]
        ai.append(ids);aa.append(at);al.append([l if a==1 else -100 for l,a in zip(ids,at)])
    ds=Dataset.from_dict({"input_ids":ai,"attention_mask":aa,"labels":al})
    ds.set_format("torch")
    loader=DataLoader(ds, batch_size=4, shuffle=True)
    opt=torch.optim.AdamW(mdl.parameters(), lr=2e-4, weight_decay=0.01)
    sched=get_cosine_schedule_with_warmup(opt, 10, (len(loader)//8)*3)

    mdl.train(); t0=time.time()
    for ep in range(3):
        tl=0;opt.zero_grad()
        for st,b in enumerate(loader):
            b={k:v.to("cuda") for k,v in b.items()}
            loss=mdl(**b).loss/8;loss.backward();tl+=loss.item()*8
            if (st+1)%8==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                opt.step();sched.step();opt.zero_grad()
        print(f"  Epoch {ep+1}/3 loss={tl/len(loader):.4f}")
    sft_path="/workspace/sft_adapter_rem"; mdl.save_pretrained(sft_path)
    print(f"  SFT done: {(time.time()-t0)/60:.1f}m")

    # Eval SFT to get error pairs
    mdl.eval(); ps,gs,errs=[],[],[]
    for i,d in enumerate(salad_test[:500]):
        c=d["conversations"]; gc=parse(c[2]["value"])
        msgs=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=512).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        pc=parse(resp); ps.append(pc); gs.append(gc)
        if pc!=gc: errs.append({"prompt":c[1]["value"],"chosen":c[2]["value"],"rejected":resp})
        if (i+1)%100==0: print(f"  eval {i+1}/500")
    del mdl; torch.cuda.empty_cache()

    # Build DPO pairs
    pairs=list(errs[:500])
    for ex in salad_train[:100]:
        c=ex["conversations"]; g=c[2]["value"]; cat=parse(g)
        w=g.replace(cat, random.choice([x for x in CATS if x!=cat]))
        pairs.append({"prompt":c[1]["value"],"chosen":g,"rejected":w})
    random.shuffle(pairs)
    print(f"  {len(pairs)} DPO pairs")

    # DPO sweep for missing seeds
    from trl import DPOTrainer, DPOConfig
    results = []
    for beta in [0.01, 0.05, 0.1, 0.5]:
        for seed in [456, 999]:
            random.seed(seed); torch.manual_seed(seed)
            print(f"\n{'='*60}\n🎯 DPO β={beta} seed={seed}\n{'='*60}")
            tok2=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
            tok2.pad_token=tok2.eos_token; tok2.padding_side="left"
            base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,
                device_map="auto",trust_remote_code=True)
            mdl2=PeftModel.from_pretrained(base,sft_path); mdl2=mdl2.merge_and_unload()
            mdl2.enable_input_require_grads()
            mdl2=get_peft_model(mdl2, LoraConfig(r=16,lora_alpha=32,
                target_modules=["q_proj","k_proj","v_proj","o_proj"],
                lora_dropout=0,bias="none",task_type="CAUSAL_LM"))
            args=DPOConfig(output_dir=f"/workspace/dpo_b{beta}_s{seed}",
                beta=beta,num_train_epochs=1,per_device_train_batch_size=2,
                gradient_accumulation_steps=8,learning_rate=5e-5,seed=seed,
                bf16=True,logging_steps=10,save_strategy="no",report_to="none",
                remove_unused_columns=False,max_length=512,max_prompt_length=384)
            t0=time.time()
            try:
                trainer=DPOTrainer(model=mdl2,args=args,
                    train_dataset=Dataset.from_list(pairs),processing_class=tok2)
                trainer.train()
            except Exception as e: print(f"  ⚠️ {e}")
            tm=(time.time()-t0)/60

            # Eval
            mdl2.eval(); ps2,gs2,errs2=[],[],[]
            for i,d in enumerate(salad_test[:500]):
                c=d["conversations"]; gc=parse(c[2]["value"])
                msgs=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]}]
                p=tok2.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                inp=tok2(p,return_tensors="pt",truncation=True,max_length=512).to("cuda")
                with torch.no_grad():
                    out=mdl2.generate(**inp,max_new_tokens=100,do_sample=False)
                resp=tok2.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
                pc=parse(resp); ps2.append(pc); gs2.append(gc)
                if pc!=gc: errs2.append(1)
                if (i+1)%100==0: print(f"  eval {i+1}/500")
            f1=f1_score(gs2,ps2,average="macro",zero_division=0)
            acc=sum(p==g for p,g in zip(ps2,gs2))/len(gs2)
            r={"experiment":"P9_DPO","beta":beta,"seed":seed,"strict_f1":round(f1,4),
               "accuracy":round(acc,4),"errors":len(errs2),"train_min":round(tm,1),
               "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
            autosave(r, f"dpo_b{beta}_s{seed}.json")
            results.append(r)
            del mdl2,base,tok2; torch.cuda.empty_cache()
    return results

# ============================================================
# RUN ALL
# ============================================================
ALL = []
print("🔴"*30+"\nPART 1: P8 LedGAR missing seeds")
for s in [456, 999]:
    ALL.append(run_p8_ledgar(s))

print("🔴"*30+"\nPART 2: P6 Scaling 5K missing seeds")
for s in [42, 77, 123]:
    ALL.append(run_p6_seed(s))

print("🔴"*30+"\nPART 3: P9 DPO missing seeds 456, 999")
ALL.extend(run_p9_dpo_remaining())

print("\n"+"="*70+"\n📊 ALL REMAINING RESULTS\n"+"="*70)
for r in ALL:
    p = r.get("paper", r.get("experiment","?"))
    d = r.get("dataset", f"β={r.get('beta','?')}")
    print(f"  {p:<12} {d:<12} s={r['seed']:<4} F1={r['strict_f1']}")
print(f"\nTotal: {len(ALL)} experiments")
print(f"Saved: {SAVE}/")
print("🏁 ALL REMAINING DONE")

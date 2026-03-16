#!/usr/bin/env python3
"""P9 ORPO — Manual implementation (no ORPOTrainer needed)
ORPO = SFT loss + λ * log_odds_ratio penalty (single model, no ref model)
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "peft>=0.13", "accelerate>=0.34",
    "datasets", "scikit-learn", "bitsandbytes"])

import torch, random, json, time, os, csv, torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import get_cosine_schedule_with_warmup

SAVE = "/workspace/results/orpo"
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

class ORPODataset(TorchDataset):
    def __init__(self, chosen_ids, chosen_mask, chosen_labels,
                 rejected_ids, rejected_mask, rejected_labels):
        self.c_ids = chosen_ids
        self.c_mask = chosen_mask
        self.c_labels = chosen_labels
        self.r_ids = rejected_ids
        self.r_mask = rejected_mask
        self.r_labels = rejected_labels

    def __len__(self): return len(self.c_ids)

    def __getitem__(self, idx):
        return {
            "chosen_ids": self.c_ids[idx], "chosen_mask": self.c_mask[idx],
            "chosen_labels": self.c_labels[idx],
            "rejected_ids": self.r_ids[idx], "rejected_mask": self.r_mask[idx],
            "rejected_labels": self.r_labels[idx],
        }

def compute_log_probs(mdl, input_ids, attention_mask, labels):
    """Compute per-token avg log probability for labeled tokens."""
    out = mdl(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]  # shift
    labs = labels[:, 1:]  # shift
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labs.unsqueeze(2).clamp(min=0)).squeeze(2)
    mask = (labs != -100).float()
    avg_log_prob = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return avg_log_prob

def orpo_loss(mdl, batch, lam=1.0):
    """ORPO loss = NLL(chosen) + λ * log_sigmoid(log_odds_ratio)"""
    # SFT loss on chosen
    c_out = mdl(input_ids=batch["chosen_ids"], attention_mask=batch["chosen_mask"],
                labels=batch["chosen_labels"])
    sft_loss = c_out.loss

    # Log probs for chosen and rejected
    c_lp = compute_log_probs(mdl, batch["chosen_ids"], batch["chosen_mask"], batch["chosen_labels"])
    r_lp = compute_log_probs(mdl, batch["rejected_ids"], batch["rejected_mask"], batch["rejected_labels"])

    # Log odds ratio: log(p_c/(1-p_c)) - log(p_r/(1-p_r))
    # = log_p_c - log(1-exp(log_p_c)) - log_p_r + log(1-exp(log_p_r))
    log_odds = (c_lp - torch.log1p(-c_lp.exp().clamp(max=0.9999))) - \
               (r_lp - torch.log1p(-r_lp.exp().clamp(max=0.9999)))

    orpo_penalty = -F.logsigmoid(log_odds).mean()
    total = sft_loss + lam * orpo_penalty
    return total, sft_loss.item(), orpo_penalty.item()

def run_orpo(seed, lam=1.0, epochs=3, lr=2e-4, batch=2, gacc=16, rank=64, ml=512):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"📊 P9 ORPO | λ={lam} | s={seed}")
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

    # Build preference pairs from training data
    random.seed(seed)
    c_ids,c_mask,c_lab = [],[],[]
    r_ids,r_mask,r_lab = [],[],[]
    for d in salad_train:
        c = d["conversations"]
        gold = c[2]["value"]; cat = parse(gold)
        # chosen = correct
        msgs_c = [{"role":"system","content":c[0]["value"]},
                   {"role":"user","content":c[1]["value"]},
                   {"role":"assistant","content":gold}]
        # rejected = wrong category
        wrong_cat = random.choice([x for x in CATS if x != cat])
        wrong = gold.replace(cat, wrong_cat)
        msgs_r = [{"role":"system","content":c[0]["value"]},
                   {"role":"user","content":c[1]["value"]},
                   {"role":"assistant","content":wrong}]

        sc = tok.apply_chat_template(msgs_c, tokenize=False)
        sr = tok.apply_chat_template(msgs_r, tokenize=False)
        ec = tok(sc, truncation=True, max_length=ml, padding="max_length")
        er = tok(sr, truncation=True, max_length=ml, padding="max_length")

        ci,ca = ec["input_ids"],ec["attention_mask"]
        ri,ra = er["input_ids"],er["attention_mask"]
        c_ids.append(ci); c_mask.append(ca)
        c_lab.append([l if a==1 else -100 for l,a in zip(ci,ca)])
        r_ids.append(ri); r_mask.append(ra)
        r_lab.append([l if a==1 else -100 for l,a in zip(ri,ra)])

    ds = ORPODataset(
        torch.tensor(c_ids), torch.tensor(c_mask), torch.tensor(c_lab),
        torch.tensor(r_ids), torch.tensor(r_mask), torch.tensor(r_lab))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 10, (len(loader)//gacc)*epochs)

    mdl.train(); t0 = time.time()
    for ep in range(epochs):
        tl,ts,to = 0,0,0; opt.zero_grad()
        for st,b in enumerate(loader):
            b = {k:v.to("cuda") for k,v in b.items()}
            loss,sl,ol = orpo_loss(mdl, b, lam=lam)
            (loss/gacc).backward()
            tl += loss.item(); ts += sl; to += ol
            if (st+1)%gacc==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        n = len(loader)
        print(f"  Epoch {ep+1}/{epochs} total={tl/n:.4f} sft={ts/n:.4f} orpo={to/n:.4f}")
    tm = (time.time()-t0)/60

    # Eval
    mdl.eval(); ps,gs = [],[]
    for i,d in enumerate(salad_test[:500]):
        c=d["conversations"]; gc=parse(c[2]["value"])
        msgs=[{"role":"system","content":c[0]["value"]},{"role":"user","content":c[1]["value"]}]
        p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inp=tok(p,return_tensors="pt",truncation=True,max_length=ml).to("cuda")
        with torch.no_grad():
            out=mdl.generate(**inp,max_new_tokens=100,do_sample=False)
        resp=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        ps.append(parse(resp)); gs.append(gc)
        if (i+1)%100==0: print(f"  eval {i+1}/500")

    f1=f1_score(gs,ps,average="macro",zero_division=0)
    acc=sum(p==g for p,g in zip(ps,gs))/len(gs)
    errs=sum(p!=g for p,g in zip(ps,gs))
    r={"experiment":"P9_ORPO","lambda":lam,"seed":seed,"strict_f1":round(f1,4),
       "accuracy":round(acc,4),"errors":errs,"train_min":round(tm,1),
       "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
    autosave(r, f"orpo_l{lam}_s{seed}.json")
    del mdl,tok; torch.cuda.empty_cache()
    return r

# ============================================================
# RUN: ORPO λ sweep × 5 seeds
# ============================================================
ALL = []
for lam in [0.1, 0.5, 1.0]:
    for seed in [42, 77, 123, 456, 999]:
        ALL.append(run_orpo(seed, lam=lam))

print("\n"+"="*70+"\n📊 P9 ORPO RESULTS\n"+"="*70)
for r in ALL:
    print(f"  λ={r['lambda']:<5} s={r['seed']:<4} F1={r['strict_f1']} acc={r['accuracy']}")
print(f"\nTotal: {len(ALL)} experiments")
print(f"Saved: {SAVE}/")
print("🏁 P9 ORPO DONE")

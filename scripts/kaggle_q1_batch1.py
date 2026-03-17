#!/usr/bin/env python3
"""
Q1 Kaggle Notebook — Seeds + P8 Experiments
============================================
Run on Kaggle with GPU T4×2. Autosaves after every experiment.
Copy-paste into a Kaggle notebook cell.

Covers:
  1. P6/P15 seed expansion (Qwen3.5-0.8B on SALAD)
  2. P20/P21 seed expansion (Qwen2.5-0.5B on AG News)
  3. P8 TBD: LLM on AG News + GoEmotions (5 seeds each)
"""

# ============================================================
# Cell 1: Setup (run this cell FIRST)
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "peft>=0.13",
    "accelerate>=0.34", "datasets", "scikit-learn"])
# NOTE: No bitsandbytes needed — 0.5B model fits in T4 16GB as float16

import torch, random, math, json, time, os, csv
from collections import Counter
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# === AUTOSAVE CONFIG ===
SAVE_DIR = "/kaggle/working/q1_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def autosave(result, filename):
    """Save result dict to JSON + append to CSV summary. Called after EVERY experiment."""
    # Save individual result
    path = os.path.join(SAVE_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    
    # Append to running CSV summary (survives crashes)
    csv_path = os.path.join(SAVE_DIR, "summary.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(result)
    
    print(f"✅ AUTOSAVED: {path}")
    print(f"📊 {result.get('paper','?')} | {result.get('domain','?')} | seed={result.get('seed','?')} | F1={result.get('strict_f1','?')} | Acc={result.get('accuracy','?')}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)


# ============================================================
# Cell 2: Cross-domain training function
# ============================================================
def train_and_eval(model_name, domain, seed, train_size=5000, test_size=1000,
                   epochs=3, lr=2e-4, batch=4, grad_acc=8, rank=64, max_len=512,
                   paper="?"):
    """Train QLoRA model on domain, evaluate, autosave results."""
    
    random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*60}")
    print(f"🚀 {paper}: {domain} seed={seed} model={model_name.split('/')[-1]}")
    print(f"{'='*60}")
    
    # --- Dataset ---
    if domain == "ag_news":
        ds = load_dataset("ag_news")
        LABELS = ["World","Sports","Business","Sci/Tech"]
        SYS = "Classify this news article into one category. Reply with ONLY the category name."
        def get_tl(ex): return ex["text"], LABELS[ex["label"]]
        tr_s, te_s = "train", "test"
    elif domain == "go_emotions":
        ds = load_dataset("google-research-datasets/go_emotions","simplified")
        LABELS = ["admiration","amusement","anger","annoyance","approval","caring","confusion",
                  "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
                  "excitement","fear","gratitude","grief","joy","love","nervousness","neutral",
                  "optimism","pride","realization","relief","remorse","sadness","surprise"]
        SYS = "Identify the primary emotion in this text. Reply with ONLY the emotion name."
        def get_tl(ex): return ex["text"], LABELS[ex["labels"][0]]
        tr_s, te_s = "train", "test"
    
    raw_tr = ds[tr_s].shuffle(seed=seed).select(range(min(train_size, len(ds[tr_s]))))
    raw_te = ds[te_s].shuffle(seed=seed).select(range(min(test_size, len(ds[te_s]))))
    lc = Counter([get_tl(ex)[1] for ex in raw_tr]); tot = sum(lc.values())
    H = -sum((c/tot)*math.log2(c/tot) for c in lc.values())
    
    # --- Model ---
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    
    mdl = AutoModelForCausalLM.from_pretrained(model_name,
        dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    mdl.print_trainable_parameters()
    
    # --- Tokenize ---
    def tokenize(ex):
        txt, lbl = get_tl(ex)
        msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]},{"role":"assistant","content":lbl}]
        s = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(s, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0); attn = enc["attention_mask"].squeeze(0)
        labels = ids.clone(); labels[attn == 0] = -100
        return {"input_ids": ids.tolist(), "attention_mask": attn.tolist(), "lm_labels": labels.tolist()}
    
    train_ds = raw_tr.map(tokenize, remove_columns=raw_tr.column_names)
    train_ds = train_ds.rename_column("lm_labels", "labels")
    train_ds.set_format("torch")
    
    # --- Train ---
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(loader) // grad_acc) * epochs
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=total_steps)
    
    mdl.train(); t0 = time.time()
    for epoch in range(epochs):
        total_loss = 0; opt.zero_grad()
        for step, b in enumerate(loader):
            b = {k: v.to("cuda") for k, v in b.items()}
            loss = mdl(**b).loss / grad_acc
            loss.backward()
            total_loss += loss.item() * grad_acc
            if (step + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"  Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")
    
    train_min = (time.time() - t0) / 60
    
    # --- Evaluate ---
    mdl.eval()
    preds, golds = [], []
    for i, ex in enumerate(raw_te):
        txt, lbl = get_tl(ex)
        msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=32, do_sample=False)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append(resp); golds.append(lbl)
        if (i + 1) % 200 == 0:
            print(f"  eval {i+1}/{len(raw_te)} f1={f1_score(golds, preds, average='macro', zero_division=0):.4f}")
    
    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    acc = sum(p == l for p, l in zip(preds, golds)) / len(golds)
    
    # --- Autosave ---
    result = {
        "paper": paper, "domain": domain, "seed": seed,
        "entropy": round(H, 2), "num_classes": len(LABELS),
        "strict_f1": round(f1, 4), "accuracy": round(acc, 4),
        "train_samples": len(raw_tr), "test_samples": len(raw_te),
        "train_time_min": round(train_min, 1),
        "model": model_name.split("/")[-1], "lora_rank": rank,
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": datetime.now().isoformat()
    }
    autosave(result, f"{paper}_{domain}_{model_name.split('/')[-1]}_s{seed}.json")
    
    # Free GPU memory
    del mdl, tok, train_ds, loader
    torch.cuda.empty_cache()
    
    return result


# ============================================================
# Cell 3: RUN ALL EXPERIMENTS
# ============================================================

ALL_RESULTS = []

# --- P20/P21: Qwen2.5-0.5B on AG News, seed 999 ---
print("\n" + "🔵"*30)
print("P20/P21: Adding seed 999")
ALL_RESULTS.append(train_and_eval(
    "Qwen/Qwen2.5-0.5B-Instruct", "ag_news", seed=999, paper="P20_P21"))

# --- P8: Qwen2.5-0.5B on AG News, 5 seeds ---
for seed in [42, 77, 123, 456, 999]:
    print("\n" + "🟢"*30)
    print(f"P8: AG News seed={seed}")
    ALL_RESULTS.append(train_and_eval(
        "Qwen/Qwen2.5-0.5B-Instruct", "ag_news", seed=seed, paper="P8"))

# --- P8: Qwen2.5-0.5B on GoEmotions, 5 seeds ---
for seed in [42, 77, 123, 456, 999]:
    print("\n" + "🟡"*30)
    print(f"P8: GoEmotions seed={seed}")
    ALL_RESULTS.append(train_and_eval(
        "Qwen/Qwen2.5-0.5B-Instruct", "go_emotions", seed=seed, paper="P8"))

# ============================================================
# Cell 4: Final Summary
# ============================================================
print("\n\n" + "="*80)
print("📊 FINAL RESULTS SUMMARY")
print("="*80)
for r in ALL_RESULTS:
    print(f"  {r['paper']:8s} | {r['domain']:15s} | seed={r['seed']:3d} | F1={r['strict_f1']:.4f} | Acc={r['accuracy']:.4f} | {r['train_time_min']:.0f}min")
print(f"\nAll {len(ALL_RESULTS)} results saved to: {SAVE_DIR}/")
print(f"CSV summary: {SAVE_DIR}/summary.csv")

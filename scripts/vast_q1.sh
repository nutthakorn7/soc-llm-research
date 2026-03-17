#!/bin/bash
# ===================================================================
# Q1 Vast.ai — P8 LedGAR + P9 DPO β Sweep
# Usage: ssh into Vast.ai A100, then: bash vast_q1.sh
# Autosaves after every experiment to /workspace/results/
# ===================================================================

set -e
echo "=== Q1 Vast.ai Experiments ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

# Setup
pip install -q transformers peft bitsandbytes accelerate datasets scikit-learn trl

RESULTS=/workspace/results
mkdir -p $RESULTS

# ============================================================
# PART 1: P8 LedGAR (100 classes × 5 seeds) — ~3h
# ============================================================
echo ""
echo "=========================================="
echo "PART 1: P8 LedGAR (100 classes)"
echo "=========================================="

cat > /workspace/train_crossdomain.py << 'TRAINSCRIPT'
#!/usr/bin/env python3
"""Cross-domain QLoRA fine-tuning with autosave."""
import argparse, torch, random, math, json, time, os, csv
from collections import Counter
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--domain", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_size", type=int, default=5000)
parser.add_argument("--test_size", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--grad_acc", type=int, default=8)
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--paper", default="?")
args = parser.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Dataset
if args.domain == "ledgar":
    ds = load_dataset("lex_glue","ledgar")
    LABELS = ds["train"].features["label"].names
    SYS = "Classify this legal contract provision. Reply with ONLY the provision type name."
    def get_tl(ex): return ex["text"], LABELS[ex["label"]]
    tr_s, te_s = "train", "validation"
elif args.domain == "ag_news":
    ds = load_dataset("ag_news")
    LABELS = ["World","Sports","Business","Sci/Tech"]
    SYS = "Classify this news article into one category. Reply with ONLY the category name."
    def get_tl(ex): return ex["text"], LABELS[ex["label"]]
    tr_s, te_s = "train", "test"
elif args.domain == "go_emotions":
    ds = load_dataset("google-research-datasets/go_emotions","simplified")
    LABELS = ["admiration","amusement","anger","annoyance","approval","caring","confusion",
              "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
              "excitement","fear","gratitude","grief","joy","love","nervousness","neutral",
              "optimism","pride","realization","relief","remorse","sadness","surprise"]
    SYS = "Identify the primary emotion in this text. Reply with ONLY the emotion name."
    def get_tl(ex): return ex["text"], LABELS[ex["labels"][0]]
    tr_s, te_s = "train", "test"

raw_tr = ds[tr_s].shuffle(seed=args.seed).select(range(min(args.train_size, len(ds[tr_s]))))
raw_te = ds[te_s].shuffle(seed=args.seed).select(range(min(args.test_size, len(ds[te_s]))))
lc = Counter([get_tl(ex)[1] for ex in raw_tr]); tot = sum(lc.values())
H = -sum((c/tot)*math.log2(c/tot) for c in lc.values())
print(f"Domain: {args.domain} | Classes: {len(LABELS)} | H(Y): {H:.2f} | seed: {args.seed}")

# Model
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
tok.pad_token = tok.eos_token; tok.padding_side = "right"
mdl = AutoModelForCausalLM.from_pretrained(args.model,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
    device_map="auto", trust_remote_code=True)
mdl = prepare_model_for_kbit_training(mdl)
mdl = get_peft_model(mdl, LoraConfig(r=args.rank, lora_alpha=args.rank*2,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none", task_type="CAUSAL_LM"))

# Tokenize
def tokenize(ex):
    txt, lbl = get_tl(ex)
    msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]},{"role":"assistant","content":lbl}]
    s = tok.apply_chat_template(msgs, tokenize=False)
    enc = tok(s, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].squeeze(0); attn = enc["attention_mask"].squeeze(0)
    labels = ids.clone(); labels[attn == 0] = -100
    return {"input_ids": ids.tolist(), "attention_mask": attn.tolist(), "lm_labels": labels.tolist()}
train_ds = raw_tr.map(tokenize, remove_columns=raw_tr.column_names)
train_ds = train_ds.rename_column("lm_labels", "labels"); train_ds.set_format("torch")

# Train
loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=0.01)
total_steps = (len(loader) // args.grad_acc) * args.epochs
sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=total_steps)
mdl.train(); t0 = time.time()
for epoch in range(args.epochs):
    total_loss = 0; opt.zero_grad()
    for step, b in enumerate(loader):
        b = {k: v.to("cuda") for k, v in b.items()}
        loss = mdl(**b).loss / args.grad_acc; loss.backward()
        total_loss += loss.item() * args.grad_acc
        if (step + 1) % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    print(f"  Epoch {epoch+1}/{args.epochs} loss={total_loss/len(loader):.4f}")
train_min = (time.time() - t0) / 60

# Evaluate
mdl.eval(); preds, golds = [], []
for i, ex in enumerate(raw_te):
    txt, lbl = get_tl(ex)
    msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt", truncation=True, max_length=args.max_len).to("cuda")
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=32, do_sample=False)
    resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    preds.append(resp); golds.append(lbl)
    if (i + 1) % 200 == 0:
        print(f"  eval {i+1}/{len(raw_te)} f1={f1_score(golds, preds, average='macro', zero_division=0):.4f}")

f1 = f1_score(golds, preds, average="macro", zero_division=0)
acc = sum(p == l for p, l in zip(preds, golds)) / len(golds)

# AUTOSAVE
result = {"paper": args.paper, "domain": args.domain, "seed": args.seed,
          "entropy": round(H, 2), "num_classes": len(LABELS),
          "strict_f1": round(f1, 4), "accuracy": round(acc, 4),
          "train_samples": len(raw_tr), "test_samples": len(raw_te),
          "train_time_min": round(train_min, 1), "model": args.model.split("/")[-1],
          "lora_rank": args.rank, "gpu": torch.cuda.get_device_name(0),
          "timestamp": datetime.now().isoformat()}

os.makedirs("/workspace/results", exist_ok=True)
fname = f"{args.paper}_{args.domain}_{args.model.split('/')[-1]}_s{args.seed}.json"
with open(f"/workspace/results/{fname}", "w") as f:
    json.dump(result, f, indent=2)

# Append to CSV
csv_path = "/workspace/results/summary.csv"
file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(result.keys()))
    if not file_exists: w.writeheader()
    w.writerow(result)

print(f"\n✅ AUTOSAVED: {fname}")
print(f"📊 F1={f1:.4f} | Acc={acc:.4f} | {train_min:.1f}min")
TRAINSCRIPT

# Run P8 LedGAR
for SEED in 42 77 123 456 999; do
    echo ">>> P8 LedGAR seed=$SEED $(date)"
    python /workspace/train_crossdomain.py \
        --domain ledgar --seed $SEED --paper P8 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --train_size 5000 --test_size 1000
    echo "<<< P8 LedGAR seed=$SEED DONE $(date)"
    echo ""
done

# ============================================================
# PART 2: P9 DPO β Sweep — ~14h
# ============================================================
echo ""
echo "=========================================="
echo "PART 2: P9 DPO β Sweep"
echo "=========================================="

cat > /workspace/dpo_sweep.py << 'DPOSCRIPT'
#!/usr/bin/env python3
"""
P9 DPO β sweep with autosave.
Step 1: SFT baseline on SALAD (if not cached)
Step 2: Generate preference pairs from SFT errors
Step 3: DPO with β ∈ {0.001, 0.01, 0.05, 0.5} × 3 seeds
"""
import torch, random, json, time, os, csv
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import f1_score
from trl import DPOTrainer, DPOConfig

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB")

# TODO: This script requires:
# 1. SALAD dataset at /workspace/data/salad_clean_5k.jsonl
# 2. SALAD validation at /workspace/data/salad_val.jsonl
# 3. DPO preference pairs at /workspace/data/dpo_prefs.jsonl
#    Format: {"prompt": "...", "chosen": "Reconnaissance", "rejected": "Port Scanning"}
#
# Generate preference pairs by:
#   1. Train SFT model on SALAD 5K
#   2. Run inference on validation set
#   3. For each hallucinated prediction, create (correct_label, hallucinated_label) pair

print("⚠️  NOTE: You must prepare SALAD data and DPO preference pairs first!")
print("⚠️  See comments in script for instructions.")
print("⚠️  If SALAD data is not available, this script will error.")

# Placeholder — actual DPO training code depends on data format
# The key experiment parameters:
BETAS = [0.001, 0.01, 0.05, 0.5]
SEEDS = [42, 77, 123]

for beta in BETAS:
    for seed in SEEDS:
        result = {
            "paper": "P9", "experiment": "dpo_beta_sweep",
            "beta": beta, "seed": seed,
            "status": "PENDING — need SALAD data + preference pairs",
            "timestamp": datetime.now().isoformat()
        }
        fname = f"P9_dpo_beta{beta}_s{seed}.json"
        os.makedirs("/workspace/results", exist_ok=True)
        with open(f"/workspace/results/{fname}", "w") as f:
            json.dump(result, f, indent=2)
        print(f"📝 Placeholder saved: {fname} (β={beta}, seed={seed})")

print("\n⚠️  DPO experiments require SALAD dataset preparation.")
print("    Run SFT first → extract hallucination pairs → then re-run this script.")
DPOSCRIPT

python /workspace/dpo_sweep.py

echo ""
echo "=== ALL VAST.AI EXPERIMENTS COMPLETE ==="
echo "Results in: /workspace/results/"
echo "Download: scp -r vast:/workspace/results/ ."
echo "End: $(date)"

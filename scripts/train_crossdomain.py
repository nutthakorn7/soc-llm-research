#!/usr/bin/env python3
"""Cross-domain QLoRA fine-tuning — works with ANY trl version.
Uses raw PyTorch loop (same technique as SFTTrainer internally).
"""
import argparse, torch, random, math, json, time, os
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--domain", required=True, choices=["ag_news","go_emotions","ledgar"])
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
args = parser.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Domain: {args.domain} | Model: {args.model}")

# === Dataset ===
if args.domain == "ag_news":
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
elif args.domain == "ledgar":
    ds = load_dataset("lex_glue","ledgar")
    LABELS = ds["train"].features["label"].names
    SYS = "Classify this legal contract provision. Reply with ONLY the provision type name."
    def get_tl(ex): return ex["text"], LABELS[ex["label"]]
    tr_s, te_s = "train", "validation"

raw_tr = ds[tr_s].shuffle(seed=args.seed).select(range(min(args.train_size, len(ds[tr_s]))))
raw_te = ds[te_s].shuffle(seed=args.seed).select(range(min(args.test_size, len(ds[te_s]))))
lc = Counter([get_tl(ex)[1] for ex in raw_tr]); tot = sum(lc.values())
H = -sum((c/tot)*math.log2(c/tot) for c in lc.values())
print(f"Classes: {len(LABELS)} | H(Y): {H:.2f} bits | Train: {len(raw_tr)} | Test: {len(raw_te)}")

# === Model + QLoRA ===
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
mdl.print_trainable_parameters()

# === Tokenize ===
def tokenize(ex):
    txt, lbl = get_tl(ex)
    msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]},{"role":"assistant","content":lbl}]
    s = tok.apply_chat_template(msgs, tokenize=False)
    enc = tok(s, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].squeeze(0); attn = enc["attention_mask"].squeeze(0)
    labels = ids.clone(); labels[attn == 0] = -100
    # Use 'lm_labels' to avoid ClassLabel schema conflict
    # (GoEmotions has a "labels" column typed ClassLabel(28) — token IDs like 151645 crash if
    #  datasets tries to cast them to that schema during .map() writes)
    return {"input_ids": ids.tolist(), "attention_mask": attn.tolist(), "lm_labels": labels.tolist()}

train_ds = raw_tr.map(tokenize, remove_columns=raw_tr.column_names)
train_ds = train_ds.rename_column("lm_labels", "labels")
train_ds.set_format("torch")
print(f"Tokenized {len(train_ds)} samples")

# === Train (PyTorch loop — same as SFTTrainer internally) ===
from transformers import get_cosine_schedule_with_warmup
loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=0.01)
total_steps = (len(loader) // args.grad_acc) * args.epochs
sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=total_steps)

mdl.train(); t0 = time.time()
for epoch in range(args.epochs):
    total_loss = 0; opt.zero_grad()
    for step, batch in enumerate(loader):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        loss = mdl(**batch).loss / args.grad_acc
        loss.backward()
        total_loss += loss.item() * args.grad_acc
        if (step + 1) % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if (step + 1) % 100 == 0:
            print(f"  E{epoch+1} step {step+1}/{len(loader)} loss={total_loss/(step+1):.4f}")
    print(f"Epoch {epoch+1}/{args.epochs} avg_loss={total_loss/len(loader):.4f}")

train_min = (time.time() - t0) / 60
print(f"\nTraining done in {train_min:.1f} min")

# === Evaluate ===
mdl.eval()
preds, golds, raws = [], [], []
for i, ex in enumerate(raw_te):
    txt, lbl = get_tl(ex)
    msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt", truncation=True, max_length=args.max_len).to("cuda")
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=32, do_sample=False)
    resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    preds.append(resp); golds.append(lbl)
    raws.append({"label": lbl, "predict": resp})
    if (i + 1) % 200 == 0:
        f1_now = f1_score(golds, preds, average="macro", zero_division=0)
        print(f"  eval {i+1}/{len(raw_te)} f1={f1_now:.4f}")

f1 = f1_score(golds, preds, average="macro", zero_division=0)
acc = sum(p == l for p, l in zip(preds, golds)) / len(golds)

print(f"\n{'='*60}")
print(f"  {args.domain} | H(Y)={H:.2f} | F1={f1:.4f} | Acc={acc:.4f}")
print(f"  Model: {args.model} | LoRA r={args.rank} | {train_min:.1f} min")
print(f"{'='*60}")

os.makedirs("/workspace/results", exist_ok=True)
res = {"domain": args.domain, "seed": args.seed, "entropy": round(H, 2),
       "strict_macro_f1": round(f1, 4), "accuracy": round(acc, 4),
       "train_samples": len(raw_tr), "test_samples": len(raw_te),
       "num_classes": len(LABELS), "train_time_min": round(train_min, 1),
       "model": args.model, "lora_rank": args.rank}
with open(f"/workspace/results/{args.domain}_{args.model.split('/')[-1]}_s{args.seed}.json", "w") as f:
    json.dump(res, f, indent=2)
with open(f"/workspace/results/{args.domain}_{args.model.split('/')[-1]}_s{args.seed}_preds.jsonl", "w") as f:
    for r in raws: f.write(json.dumps(r) + "\n")
print(f"Saved to /workspace/results/")

#!/usr/bin/env python3
"""
P9 DPO β Sweep — Google Colab Version
======================================
SFT → collect errors → build preference pairs → DPO β sweep
Qwen2.5-0.5B (~1GB) fits on Colab T4 easily.

Run: Runtime > Change runtime type > T4 GPU
"""

# ============================================================
# Cell 1: Setup + Mount Drive (autosave)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "peft>=0.13", "trl>=0.12",
    "accelerate>=0.34", "datasets", "scikit-learn"])

import torch, random, json, time, os, csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Save to Google Drive for persistence
SAVE_DIR = "/content/drive/MyDrive/p9_results"
os.makedirs(SAVE_DIR, exist_ok=True)
LOCAL_DIR = "/content/p9_local"
os.makedirs(LOCAL_DIR, exist_ok=True)

ATTACK_CATS = ["Analysis", "Backdoor", "DoS", "Exploits",
               "Fuzzers", "Generic", "Reconnaissance", "Shellcode"]

def autosave(result, filename):
    # Save locally + Drive
    for d in [LOCAL_DIR, SAVE_DIR]:
        with open(os.path.join(d, filename), "w") as f:
            json.dump(result, f, indent=2)
    csv_path = os.path.join(SAVE_DIR, "summary.csv")
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not exists: w.writeheader()
        w.writerow(result)
    print(f"✅ SAVED: {filename} → Drive")
    print(f"   β={result.get('beta','?')} seed={result.get('seed','?')} F1={result.get('strict_f1','?')}")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================
# Cell 2: Download SALAD Data
# ============================================================
DATA_DIR = "/content/salad_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Download from GitHub
!wget -q -O {DATA_DIR}/train_5k_clean.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/train_5k_clean.json"
!wget -q -O {DATA_DIR}/test_held_out.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/test_held_out.json"

train_data = json.load(open(f"{DATA_DIR}/train_5k_clean.json"))
test_data = json.load(open(f"{DATA_DIR}/test_held_out.json"))
print(f"✅ SALAD loaded: {len(train_data)} train, {len(test_data)} test")

def parse_response(text):
    for cat in ATTACK_CATS:
        if cat.lower() in text.lower():
            return cat
    return text.strip().split("\n")[-1].strip()

# ============================================================
# Cell 3: SFT + Eval Functions
# ============================================================
def train_sft(model_name, train_data, seed=42, epochs=3, lr=2e-4,
              batch=4, grad_acc=8, rank=64, max_len=512):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'🔧'*20}")
    print(f"SFT Baseline | seed={seed}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"

    mdl = AutoModelForCausalLM.from_pretrained(model_name,
        dtype=torch.float16, device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))

    from datasets import Dataset
    def tokenize_salad(examples):
        convs = examples["conversations"]
        all_ids, all_attn, all_labels = [], [], []
        for conv in convs:
            msgs = [{"role":"system","content":conv[0]["value"]},
                    {"role":"user","content":conv[1]["value"]},
                    {"role":"assistant","content":conv[2]["value"]}]
            s = tok.apply_chat_template(msgs, tokenize=False)
            enc = tok(s, truncation=True, max_length=max_len, padding="max_length")
            ids = enc["input_ids"]; attn = enc["attention_mask"]
            labels = [l if a==1 else -100 for l,a in zip(ids, attn)]
            all_ids.append(ids); all_attn.append(attn); all_labels.append(labels)
        return {"input_ids": all_ids, "attention_mask": all_attn, "labels": all_labels}

    ds = Dataset.from_dict({"conversations": [d["conversations"] for d in train_data]})
    ds = ds.map(tokenize_salad, batched=True, batch_size=100, remove_columns=["conversations"])
    ds.set_format("torch")

    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(loader)//grad_acc)*epochs
    sched = get_cosine_schedule_with_warmup(opt, 10, total_steps)

    mdl.train(); t0 = time.time()
    for epoch in range(epochs):
        total_loss = 0; opt.zero_grad()
        for step, b in enumerate(loader):
            b = {k:v.to("cuda") for k,v in b.items()}
            loss = mdl(**b).loss / grad_acc; loss.backward()
            total_loss += loss.item()*grad_acc
            if (step+1)%grad_acc==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"  Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")
    train_min = (time.time()-t0)/60
    print(f"  SFT done: {train_min:.1f} min")

    sft_path = f"{LOCAL_DIR}/sft_adapter"
    mdl.save_pretrained(sft_path)
    return mdl, tok, sft_path, train_min

def eval_model(mdl, tok, test_data, max_samples=500, max_len=512):
    mdl.eval(); preds, golds, errors = [], [], []
    for i, ex in enumerate(test_data[:max_samples]):
        conv = ex["conversations"]
        gold_cat = parse_response(conv[2]["value"])
        msgs = [{"role":"system","content":conv[0]["value"]},
                {"role":"user","content":conv[1]["value"]}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=100, do_sample=False)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred_cat = parse_response(resp)
        preds.append(pred_cat); golds.append(gold_cat)
        if pred_cat != gold_cat:
            errors.append({"prompt":conv[1]["value"], "gold":conv[2]["value"],
                          "pred":resp, "gold_cat":gold_cat, "pred_cat":pred_cat})
        if (i+1)%100==0:
            print(f"  eval {i+1}/{max_samples} f1={f1_score(golds,preds,average='macro',zero_division=0):.4f}")
    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    acc = sum(p==g for p,g in zip(preds,golds))/len(golds)
    return f1, acc, errors

# ============================================================
# Cell 4: Run SFT Baseline
# ============================================================
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
sft_model, sft_tok, sft_path, sft_time = train_sft(MODEL, train_data, seed=42)

sft_f1, sft_acc, sft_errors = eval_model(sft_model, sft_tok, test_data, max_samples=500)
sft_result = {
    "experiment":"P9_SFT_baseline", "beta":"N/A", "seed":42,
    "strict_f1":round(sft_f1,4), "accuracy":round(sft_acc,4),
    "errors":len(sft_errors), "train_min":round(sft_time,1),
    "model":"Qwen2.5-0.5B", "gpu":torch.cuda.get_device_name(0),
    "ts":datetime.now().isoformat()
}
autosave(sft_result, "P9_sft_baseline.json")
del sft_model, sft_tok; torch.cuda.empty_cache()

# ============================================================
# Cell 5: Build Preference Pairs
# ============================================================
def build_dpo_pairs(errors, train_data, max_pairs=500):
    pairs = []
    for err in errors[:max_pairs]:
        pairs.append({"prompt":err["prompt"], "chosen":err["gold"], "rejected":err["pred"]})
    for ex in train_data[:100]:
        conv = ex["conversations"]
        gold = conv[2]["value"]; cat = parse_response(gold)
        wrong = gold.replace(cat, random.choice([c for c in ATTACK_CATS if c!=cat]))
        pairs.append({"prompt":conv[1]["value"], "chosen":gold, "rejected":wrong})
    random.shuffle(pairs)
    print(f"✅ Built {len(pairs)} DPO pairs ({len(errors)} from errors)")
    return pairs

dpo_pairs = build_dpo_pairs(sft_errors, train_data)
with open(f"{SAVE_DIR}/dpo_pairs.json","w") as f:
    json.dump(dpo_pairs, f, indent=2)

# ============================================================
# Cell 6: DPO β Sweep
# ============================================================
def run_dpo(model_name, sft_path, dpo_pairs, beta, seed, max_len=512):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'🎯'*20}")
    print(f"DPO β={beta} seed={seed}")

    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(model_name,
        dtype=torch.float16, device_map="auto", trust_remote_code=True)
    mdl = PeftModel.from_pretrained(base, sft_path)
    mdl = mdl.merge_and_unload()
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))

    dpo_ds = Dataset.from_list(dpo_pairs)
    args = DPOConfig(
        output_dir=f"{LOCAL_DIR}/dpo_b{beta}_s{seed}",
        beta=beta, num_train_epochs=1,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=5e-5, seed=seed, bf16=True,
        logging_steps=10, save_strategy="no", report_to="none",
        remove_unused_columns=False,
        max_length=max_len, max_prompt_length=max_len-128,
    )

    t0 = time.time()
    try:
        trainer = DPOTrainer(model=mdl, args=args,
                            train_dataset=dpo_ds, processing_class=tok)
        trainer.train()
    except Exception as e:
        print(f"  ⚠️ DPO error: {e}")
    train_min = (time.time()-t0)/60

    f1, acc, errors = eval_model(mdl, tok, test_data, max_samples=500)
    result = {
        "experiment":"P9_DPO", "beta":beta, "seed":seed,
        "strict_f1":round(f1,4), "accuracy":round(acc,4),
        "errors":len(errors), "train_min":round(train_min,1),
        "model":"Qwen2.5-0.5B", "gpu":torch.cuda.get_device_name(0),
        "ts":datetime.now().isoformat()
    }
    autosave(result, f"P9_dpo_b{beta}_s{seed}.json")
    del mdl, base, tok; torch.cuda.empty_cache()
    return result

ALL_RESULTS = [sft_result]
for beta in [0.01, 0.05, 0.1, 0.5]:
    for seed in [42, 77, 123]:
        ALL_RESULTS.append(run_dpo(MODEL, sft_path, dpo_pairs, beta, seed))

# ============================================================
# Cell 7: Summary
# ============================================================
print("\n" + "="*70)
print("📊 P9 DPO β SWEEP — FINAL RESULTS")
print("="*70)
print(f"{'Experiment':<18} {'β':<7} {'Seed':<6} {'F1':<10} {'Errors'}")
print("-"*55)
for r in ALL_RESULTS:
    print(f"{r['experiment']:<18} {str(r['beta']):<7} {r['seed']:<6} {r['strict_f1']:<10} {r.get('errors','?')}")
print(f"\nAll saved to Google Drive: {SAVE_DIR}/")

#!/usr/bin/env python3
"""P18: Zero-Shot Generalization — Leave-one-out on SALAD-like multi-class task.
Uses AG News (4 classes) as proxy. Train on K-1 classes, test on held-out class.
~4 hours total (8 folds × ~30 min each).
"""
import argparse, torch, random, math, json, time, os
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42; EPOCHS = 3; LR = 2e-4; RANK = 64; MAX_LEN = 512

def run_fold(held_out_class, all_labels, ds_train, ds_test):
    random.seed(SEED); torch.manual_seed(SEED)
    train_labels = [l for l in all_labels if l != held_out_class]
    
    print(f"\n{'='*60}")
    print(f"P18 Fold: Held-out='{held_out_class}' | Train on {train_labels}")
    print(f"{'='*60}")
    
    # Filter: train on K-1, test on held-out only
    train_data = [ex for ex in ds_train if all_labels[ex["label"]] != held_out_class]
    test_data = [ex for ex in ds_test if all_labels[ex["label"]] == held_out_class]
    
    random.shuffle(train_data); train_data = train_data[:3000]
    test_data = test_data[:500]
    
    if len(test_data) < 10:
        print(f"  Skipping: only {len(test_data)} test samples")
        return None
    
    print(f"  Train: {len(train_data)} (classes: {train_labels})")
    print(f"  Test: {len(test_data)} (held-out: {held_out_class})")
    
    SYS = f"Classify this text into one category: {', '.join(train_labels)}. Reply with ONLY the category name."
    
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    
    mdl = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True)
    mdl = prepare_model_for_kbit_training(mdl)
    mdl = get_peft_model(mdl, LoraConfig(r=RANK, lora_alpha=RANK*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    
    # Tokenize
    def tokenize(ex):
        txt, lbl = ex["text"], all_labels[ex["label"]]
        msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]},{"role":"assistant","content":lbl}]
        s = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(s, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0); attn = enc["attention_mask"].squeeze(0)
        labels = ids.clone(); labels[attn == 0] = -100
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}
    
    from datasets import Dataset
    tds = Dataset.from_list(train_data).map(tokenize, remove_columns=Dataset.from_list(train_data).column_names)
    tds.set_format("torch")
    
    loader = DataLoader(tds, batch_size=4, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(loader) // 8) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=max(total_steps, 1))
    
    mdl.train(); t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0; opt.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = mdl(**batch).loss / 8; loss.backward(); total_loss += loss.item() * 8
            if (step + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"  Epoch {epoch+1} loss={total_loss/max(len(loader),1):.4f}")
    train_min = (time.time() - t0) / 60
    
    # Evaluate on held-out class
    mdl.eval(); preds, golds, vocab_dist = [], [], Counter()
    for ex in test_data:
        txt = ex["text"]
        msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=32, do_sample=False)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append(resp); golds.append(held_out_class); vocab_dist[resp] += 1
    
    correct = sum(1 for p in preds if p == held_out_class)
    acc = correct / len(preds)
    halluc_rate = sum(1 for p in preds if p not in all_labels) / len(preds)
    
    print(f"  Held-out '{held_out_class}': Acc={acc:.4f} Halluc={halluc_rate:.4f}")
    print(f"  Predicted vocab: {dict(vocab_dist.most_common(5))}")
    
    res = {"paper": "P18", "held_out": held_out_class, "accuracy": round(acc, 4),
           "hallucination_rate": round(halluc_rate, 4), "n_test": len(test_data),
           "n_train": len(train_data), "train_time_min": round(train_min, 1),
           "predicted_vocab": dict(vocab_dist.most_common(10))}
    
    del mdl, opt, loader; torch.cuda.empty_cache()
    return res

if __name__ == "__main__":
    ds = load_dataset("ag_news")
    LABELS = ["World", "Sports", "Business", "Sci/Tech"]
    train_list = list(ds["train"])
    test_list = list(ds["test"])
    
    results = []
    for cls in LABELS:
        r = run_fold(cls, LABELS, train_list, test_list)
        if r:
            results.append(r)
            os.makedirs("/workspace/results", exist_ok=True)
            with open(f"/workspace/results/p18_zeroshot_{cls.replace('/','_')}.json", "w") as f:
                json.dump(r, f, indent=2)
    
    print("\n\n=== P18 Summary ===")
    for r in results:
        print(f"  Held-out '{r['held_out']}': Acc={r['accuracy']:.4f} Halluc={r['hallucination_rate']:.4f}")

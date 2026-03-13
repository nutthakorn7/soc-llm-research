#!/usr/bin/env python3
"""P9: SFT vs DPO vs ORPO — Compare 3 alignment methods on classification.
SFT = standard fine-tuning (baseline). DPO/ORPO need preference pairs.
~3 hours total.
"""
import torch, random, json, time, os, math
from collections import Counter
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42; EPOCHS = 3; LR = 2e-4; RANK = 64; MAX_LEN = 512
TRAIN_SIZE = 5000; TEST_SIZE = 1000

# Use AG News as classification task
ds = load_dataset("ag_news")
LABELS = ["World","Sports","Business","Sci/Tech"]
SYS = "Classify this news article. Reply with ONLY the category name."

def get_tl(ex): return ex["text"], LABELS[ex["label"]]

random.seed(SEED)
raw_tr = ds["train"].shuffle(seed=SEED).select(range(TRAIN_SIZE))
raw_te = ds["test"].shuffle(seed=SEED).select(range(TEST_SIZE))

def run_sft(method_name="SFT"):
    """Standard supervised fine-tuning."""
    torch.manual_seed(SEED)
    print(f"\n{'='*60}\nP9: {method_name}\n{'='*60}")
    
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
    
    def tokenize(ex):
        txt, lbl = get_tl(ex)
        msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]},{"role":"assistant","content":lbl}]
        s = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(s, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0); attn = enc["attention_mask"].squeeze(0)
        labels = ids.clone(); labels[attn == 0] = -100
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}
    
    train_ds = raw_tr.map(tokenize, remove_columns=raw_tr.column_names)
    train_ds.set_format("torch")
    
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(loader) // 8) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=total_steps)
    
    mdl.train(); t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0; opt.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = mdl(**batch).loss / 8; loss.backward(); total_loss += loss.item() * 8
            if (step + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"  Epoch {epoch+1} loss={total_loss/len(loader):.4f}")
    train_min = (time.time() - t0) / 60
    
    # Evaluate
    mdl.eval(); preds, golds = [], []
    for i, ex in enumerate(raw_te):
        txt, lbl = get_tl(ex)
        msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=32, do_sample=False)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append(resp); golds.append(lbl)
        if (i+1) % 200 == 0: print(f"  eval {i+1}/{TEST_SIZE}")
    
    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    acc = sum(p == l for p, l in zip(preds, golds)) / len(golds)
    halluc = sum(1 for p in preds if p not in LABELS) / len(preds)
    
    print(f"  {method_name}: F1={f1:.4f} Acc={acc:.4f} Halluc={halluc:.4f}")
    
    del mdl, opt, loader; torch.cuda.empty_cache()
    return {"paper": "P9", "method": method_name, "f1": round(f1, 4),
            "accuracy": round(acc, 4), "hallucination_rate": round(halluc, 4),
            "train_time_min": round(train_min, 1)}

def run_dpo_style(method_name="DPO"):
    """Proper DPO: SFT warmup → freeze ref model → DPO with per-token log probs."""
    torch.manual_seed(SEED)
    print(f"\n{'='*60}\nP9: {method_name} (preference-based)\n{'='*60}")
    
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    
    # Step 1: SFT warmup (1 epoch) to get a reasonable starting point
    mdl = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True)
    mdl = prepare_model_for_kbit_training(mdl)
    mdl = get_peft_model(mdl, LoraConfig(r=RANK, lora_alpha=RANK*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    
    def tokenize(ex):
        txt, lbl = get_tl(ex)
        msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]},{"role":"assistant","content":lbl}]
        s = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(s, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0); attn = enc["attention_mask"].squeeze(0)
        labels = ids.clone(); labels[attn == 0] = -100
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}
    
    train_ds = raw_tr.map(tokenize, remove_columns=raw_tr.column_names)
    train_ds.set_format("torch")
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
    
    # SFT warmup: 1 epoch
    print("  [SFT warmup: 1 epoch]")
    mdl.train(); t0 = time.time()
    total_loss = 0; opt.zero_grad()
    for step, batch in enumerate(loader):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        loss = mdl(**batch).loss / 8; loss.backward(); total_loss += loss.item() * 8
        if (step + 1) % 8 == 0:
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); opt.zero_grad()
    print(f"  SFT warmup loss={total_loss/len(loader):.4f}")
    
    # Step 2: Save ref model weights (frozen copy via state_dict)
    import copy
    ref_state = copy.deepcopy({k: v.clone() for k, v in mdl.named_parameters() if v.requires_grad})
    
    # Step 3: DPO training with per-token log probs
    def get_per_token_logprob(model, input_ids, attention_mask, labels):
        """Compute sum of per-token log probabilities for the response tokens."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # shift
        target = labels[:, 1:]  # shift
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        # Only count non-padding tokens (labels != -100)
        mask = (target != -100).float()
        return (token_log_probs * mask).sum(dim=1)  # sum per sequence
    
    def get_ref_logprob(input_ids, attention_mask, labels):
        """Get log probs from reference model by temporarily loading ref weights."""
        # Save current weights, load ref, compute, restore
        current_state = {k: v.clone() for k, v in mdl.named_parameters() if v.requires_grad}
        for name, param in mdl.named_parameters():
            if name in ref_state:
                param.data.copy_(ref_state[name])
        with torch.no_grad():
            ref_lp = get_per_token_logprob(mdl, input_ids, attention_mask, labels)
        for name, param in mdl.named_parameters():
            if name in current_state:
                param.data.copy_(current_state[name])
        return ref_lp
    
    # Create preference pairs
    print("  [DPO: 2 epochs with preference pairs]")
    beta = 0.1
    opt2 = torch.optim.AdamW(mdl.parameters(), lr=LR/2, weight_decay=0.01)
    
    for epoch in range(2):
        total_loss = 0; opt2.zero_grad()
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to("cuda")
            attn_mask = batch["attention_mask"].to("cuda")
            labels_chosen = batch["labels"].to("cuda")
            
            # Create rejected: swap labels with random wrong labels
            # We do this by shifting the last non-padding tokens
            labels_reject = labels_chosen.clone()
            for b in range(labels_reject.shape[0]):
                # Find response tokens (non -100 at the end)
                resp_mask = labels_reject[b] != -100
                resp_positions = resp_mask.nonzero(as_tuple=True)[0]
                if len(resp_positions) > 0:
                    # Replace response with a random wrong label token
                    wrong_labels = [l for l in LABELS if tok.decode(labels_reject[b, resp_positions[-1]].unsqueeze(0), skip_special_tokens=True).strip() != l]
                    if wrong_labels:
                        wrong = random.choice(wrong_labels)
                        wrong_ids = tok(wrong, add_special_tokens=False)["input_ids"]
                        # Replace last N response tokens
                        for j, wid in enumerate(wrong_ids):
                            if j < len(resp_positions):
                                labels_reject[b, resp_positions[-(len(wrong_ids)-j)]] = wid
            
            # Policy log probs
            pi_chosen = get_per_token_logprob(mdl, input_ids, attn_mask, labels_chosen)
            pi_reject = get_per_token_logprob(mdl, input_ids, attn_mask, labels_reject)
            
            # Reference log probs
            ref_chosen = get_ref_logprob(input_ids, attn_mask, labels_chosen)
            ref_reject = get_ref_logprob(input_ids, attn_mask, labels_reject)
            
            # DPO loss: -log(σ(β * ((π_θ(y_w) - π_ref(y_w)) - (π_θ(y_l) - π_ref(y_l)))))
            log_ratio = (pi_chosen - ref_chosen) - (pi_reject - ref_reject)
            loss = -torch.nn.functional.logsigmoid(beta * log_ratio).mean() / 4
            loss.backward(); total_loss += loss.item() * 4
            
            if (step + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt2.step(); opt2.zero_grad()
            if (step + 1) % 200 == 0:
                print(f"  E{epoch+1} step {step+1}/{len(loader)} loss={total_loss/(step+1):.4f}")
        print(f"  DPO Epoch {epoch+1} loss={total_loss/max(len(loader),1):.4f}")
    train_min = (time.time() - t0) / 60
    
    # Evaluate
    mdl.eval(); preds, golds = [], []
    for i, ex in enumerate(raw_te):
        txt, lbl = get_tl(ex)
        msgs = [{"role":"system","content":SYS},{"role":"user","content":txt[:400]}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=32, do_sample=False)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append(resp); golds.append(lbl)
        if (i+1) % 200 == 0: print(f"  eval {i+1}/{TEST_SIZE}")
    
    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    acc = sum(p == l for p, l in zip(preds, golds)) / len(golds)
    halluc = sum(1 for p in preds if p not in LABELS) / len(preds)
    
    print(f"  {method_name}: F1={f1:.4f} Acc={acc:.4f} Halluc={halluc:.4f}")
    del mdl, opt2, loader; torch.cuda.empty_cache()
    return {"paper": "P9", "method": method_name, "f1": round(f1, 4),
            "accuracy": round(acc, 4), "hallucination_rate": round(halluc, 4),
            "train_time_min": round(train_min, 1)}

if __name__ == "__main__":
    results = []
    for method, fn in [("SFT", run_sft), ("DPO", run_dpo_style)]:
        r = fn(method)
        results.append(r)
        os.makedirs("/workspace/results", exist_ok=True)
        with open(f"/workspace/results/p9_{method.lower()}.json", "w") as f:
            json.dump(r, f, indent=2)
    
    print("\n\n=== P9 Summary ===")
    for r in results:
        print(f"  {r['method']}: F1={r['f1']:.4f} Halluc={r['hallucination_rate']:.4f}")

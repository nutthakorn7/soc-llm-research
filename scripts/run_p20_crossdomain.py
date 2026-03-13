#!/usr/bin/env python3
"""P20: Cross-Domain Transfer — Train on domain A, test on domain B.
Tests whether fine-tuned LLMs retain general capability.
2 models (0.5B, 7B) × 4 domains = 8 training runs + 32 cross-eval pairs.
~6 hours total.
"""
import torch, random, json, time, os, math, itertools
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

MAX_LEN = 512; EPOCHS = 3; LR = 2e-4; RANK = 64
TRAIN_SIZE = 3000; TEST_SIZE = 500  # Smaller for speed

DOMAINS = {
    "ag_news": {"dataset": "ag_news", "split_tr": "train", "split_te": "test",
                "labels": ["World","Sports","Business","Sci/Tech"],
                "sys": "Classify this news article. Reply with ONLY the category name.",
                "get_tl": lambda ex, L: (ex["text"], L[ex["label"]])},
    "go_emotions": {"dataset": "google-research-datasets/go_emotions", "subset": "simplified",
                    "split_tr": "train", "split_te": "test",
                    "labels": ["admiration","amusement","anger","annoyance","approval","caring","confusion",
                               "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
                               "excitement","fear","gratitude","grief","joy","love","nervousness","neutral",
                               "optimism","pride","realization","relief","remorse","sadness","surprise"],
                    "sys": "Identify the primary emotion. Reply with ONLY the emotion name.",
                    "get_tl": lambda ex, L: (ex["text"], L[ex["labels"][0]])},
}

def load_domain(name):
    cfg = DOMAINS[name]
    kwargs = {"path": cfg["dataset"]}
    if "subset" in cfg: kwargs["name"] = cfg["subset"]
    ds = load_dataset(**kwargs)
    return ds, cfg

def train_and_eval(train_domain, test_domain, model_name, seed=42):
    random.seed(seed); torch.manual_seed(seed)
    
    # Load train data
    ds_tr, cfg_tr = load_domain(train_domain)
    ds_te, cfg_te = load_domain(test_domain)
    
    raw_tr = ds_tr[cfg_tr["split_tr"]].shuffle(seed=seed).select(range(min(TRAIN_SIZE, len(ds_tr[cfg_tr["split_tr"]]))))
    raw_te = ds_te[cfg_te["split_te"]].shuffle(seed=seed).select(range(min(TEST_SIZE, len(ds_te[cfg_te["split_te"]]))))
    
    is_cross = train_domain != test_domain
    tag = f"{'CROSS' if is_cross else 'IN'}-domain"
    print(f"\n{'='*60}")
    print(f"P20 {tag}: Train={train_domain} → Test={test_domain} | {model_name.split('/')[-1]}")
    print(f"{'='*60}")
    
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    
    batch_size = 2 if "7B" in model_name else 4
    grad_acc = 16 if "7B" in model_name else 8
    
    mdl = AutoModelForCausalLM.from_pretrained(model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True)
    mdl = prepare_model_for_kbit_training(mdl)
    mdl = get_peft_model(mdl, LoraConfig(r=RANK, lora_alpha=RANK*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    
    SYS_TR = cfg_tr["sys"]
    get_tl_tr = cfg_tr["get_tl"]
    LABELS_TR = cfg_tr["labels"]
    
    def tokenize(ex):
        txt, lbl = get_tl_tr(ex, LABELS_TR)
        msgs = [{"role":"system","content":SYS_TR},{"role":"user","content":txt[:400]},{"role":"assistant","content":lbl}]
        s = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(s, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0); attn = enc["attention_mask"].squeeze(0)
        labels = ids.clone(); labels[attn == 0] = -100
        # Use 'lm_labels' to avoid ClassLabel schema conflict
        # (GoEmotions "labels" column is ClassLabel(28) — token IDs crash .map() writes)
        return {"input_ids": ids.tolist(), "attention_mask": attn.tolist(), "lm_labels": labels.tolist()}
    
    train_ds = raw_tr.map(tokenize, remove_columns=raw_tr.column_names)
    train_ds = train_ds.rename_column("lm_labels", "labels")
    train_ds.set_format("torch")
    
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(loader) // grad_acc) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=max(total_steps, 1))
    
    mdl.train(); t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0; opt.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = mdl(**batch).loss / grad_acc; loss.backward(); total_loss += loss.item() * grad_acc
            if (step + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"  Epoch {epoch+1} loss={total_loss/max(len(loader),1):.4f}")
    train_min = (time.time() - t0) / 60
    
    # Evaluate on TEST domain (may be different from train domain)
    SYS_TE = cfg_te["sys"]
    get_tl_te = cfg_te["get_tl"]
    LABELS_TE = cfg_te["labels"]
    
    mdl.eval(); preds, golds, vocab = [], [], Counter()
    for i, ex in enumerate(raw_te):
        txt, lbl = get_tl_te(ex, LABELS_TE)
        msgs = [{"role":"system","content":SYS_TE},{"role":"user","content":txt[:400]}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=32, do_sample=False)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append(resp); golds.append(lbl); vocab[resp] += 1
        if (i+1) % 100 == 0: print(f"  eval {i+1}/{len(raw_te)}")
    
    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    f1_lenient = f1_score([g.lower().strip() for g in golds],
                          [p.lower().strip() for p in preds],
                          average="macro", zero_division=0)
    acc = sum(p == l for p, l in zip(preds, golds)) / len(golds)
    acc_lenient = sum(p.lower().strip() == l.lower().strip() for p, l in zip(preds, golds)) / len(golds)
    
    print(f"  F1={f1:.4f} F1_lenient={f1_lenient:.4f} Acc={acc:.4f} Top vocab: {dict(vocab.most_common(5))}")
    
    del mdl, opt, loader; torch.cuda.empty_cache()
    
    res = {"paper": "P20", "train_domain": train_domain, "test_domain": test_domain,
           "is_cross_domain": is_cross, "model": model_name.split("/")[-1],
           "f1": round(f1, 4), "f1_lenient": round(f1_lenient, 4),
           "accuracy": round(acc, 4), "accuracy_lenient": round(acc_lenient, 4),
           "train_time_min": round(train_min, 1), "seed": seed,
           "top_predictions": dict(vocab.most_common(10))}
    
    os.makedirs("/workspace/results", exist_ok=True)
    fname = f"p20_{train_domain}_to_{test_domain}_{model_name.split('/')[-1]}.json"
    with open(f"/workspace/results/{fname}", "w") as f:
        json.dump(res, f, indent=2)
    return res

if __name__ == "__main__":
    results = []
    domains = list(DOMAINS.keys())
    
    # For each model, train on each domain and test on all domains
    for model_name in ["Qwen/Qwen2.5-0.5B-Instruct"]:  # 7B takes too long, skip for now
        for train_d in domains:
            for test_d in domains:
                r = train_and_eval(train_d, test_d, model_name)
                results.append(r)
    
    print("\n\n=== P20 Cross-Domain Transfer Matrix ===")
    print(f"{'Train→Test':<30} {'F1':>8}")
    for r in results:
        tag = "✓" if not r["is_cross_domain"] else "✗"
        print(f"  {tag} {r['train_domain']}→{r['test_domain']}: F1={r['f1']:.4f}")

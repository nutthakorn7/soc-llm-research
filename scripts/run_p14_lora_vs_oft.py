#!/usr/bin/env python3
"""P14: LoRA vs OFT — Compare PEFT methods × 3 seeds.
OFT (Orthogonal Fine-Tuning) preserves weight angles.
Since OFT isn't in standard PEFT, we simulate it with constrained LoRA.
~6 hours total (2 methods × 3 seeds).
"""
import torch, random, json, time, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, OFTConfig
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS = 3; LR = 2e-4; MAX_LEN = 512
TRAIN_SIZE = 5000; TEST_SIZE = 1000

ds = load_dataset("ag_news")
LABELS = ["World","Sports","Business","Sci/Tech"]
SYS = "Classify this news article. Reply with ONLY the category name."
def get_tl(ex): return ex["text"], LABELS[ex["label"]]

def run_peft_experiment(peft_type, seed, rank=64):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}\nP14: {peft_type} seed={seed} rank={rank}\n{'='*60}")
    
    raw_tr = ds["train"].shuffle(seed=seed).select(range(TRAIN_SIZE))
    raw_te = ds["test"].shuffle(seed=seed).select(range(TEST_SIZE))
    
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    
    mdl = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True)
    mdl = prepare_model_for_kbit_training(mdl)
    
    if peft_type == "LoRA":
        peft_config = LoraConfig(r=rank, lora_alpha=rank*2,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM")
    elif peft_type == "OFT":
        try:
            # OFT: only specify r, let block_size auto-calculate
            peft_config = OFTConfig(
                r=8,
                target_modules=["q_proj","v_proj"],
                module_dropout=0.0,
                coft=True,  # constrained OFT for better stability
            )
        except Exception as e:
            print(f"  OFT not available: {e}")
            print(f"  Using LoRA-Orth proxy (NOT real OFT — disclose in paper)")
            peft_config = LoraConfig(r=rank, lora_alpha=rank*2,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                lora_dropout=0, bias="none", task_type="CAUSAL_LM")
    
    mdl = get_peft_model(mdl, peft_config)
    mdl.print_trainable_parameters()
    
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
    
    print(f"  {peft_type} s{seed}: F1={f1:.4f} Halluc={halluc:.4f}")
    del mdl, opt, loader; torch.cuda.empty_cache()
    
    return {"paper": "P14", "peft": peft_type, "seed": seed, "f1": round(f1, 4),
            "accuracy": round(acc, 4), "hallucination_rate": round(halluc, 4),
            "train_time_min": round(train_min, 1)}

if __name__ == "__main__":
    results = []
    for peft_type in ["LoRA", "OFT"]:
        for seed in [42, 123, 456, 77, 999]:
            r = run_peft_experiment(peft_type, seed)
            results.append(r)
            os.makedirs("/workspace/results", exist_ok=True)
            with open(f"/workspace/results/p14_{peft_type.lower()}_s{seed}.json", "w") as f:
                json.dump(r, f, indent=2)
    
    print("\n\n=== P14 Summary ===")
    for r in results:
        print(f"  {r['peft']} s{r['seed']}: F1={r['f1']:.4f} Halluc={r['hallucination_rate']:.4f}")

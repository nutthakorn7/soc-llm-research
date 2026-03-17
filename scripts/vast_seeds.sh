#!/bin/bash
# P14 + P15 Seed Expansion — Vast.ai (runs after P9)
# ===================================================
# P14: LoRA vs OFT on AG News (2 new seeds: 456, 999)
# P15: Multi-task SALAD (cls+tri+atk) (2 new seeds: 456, 999)
# Both use Qwen2.5-0.5B → fits on 4090
#
# Usage: ssh vast 'nohup bash /workspace/vast_seeds.sh >> /workspace/seeds.log 2>&1 &'

set -e
echo "🚀 Seed Expansion (P14+P15) — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

pip install -q transformers>=4.45 peft>=0.13 accelerate>=0.34 datasets scikit-learn 2>/dev/null

mkdir -p /workspace/results/seeds
mkdir -p /workspace/salad_data

# Download data if not already present (P9 may have done this)
[ ! -f /workspace/salad_data/train_5k_clean.json ] && \
    wget -q -O /workspace/salad_data/train_5k_clean.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/train_5k_clean.json"
[ ! -f /workspace/salad_data/test_held_out.json ] && \
    wget -q -O /workspace/salad_data/test_held_out.json \
    "https://raw.githubusercontent.com/nutthakorn7/soc-llm-research/main/data/test_held_out.json"

python3 << 'PYEOF'
import torch, random, json, time, os, csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, OFTConfig, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import Dataset, load_dataset

SAVE = "/workspace/results/seeds"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def autosave(r, fn):
    with open(f"{SAVE}/{fn}","w") as f: json.dump(r,f,indent=2)
    cp = f"{SAVE}/summary.csv"; ex = os.path.exists(cp)
    with open(cp,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        if not ex: w.writeheader()
        w.writerow(r)
    print(f"✅ SAVED {fn} | {r.get('paper','')} {r.get('peft_type','')} seed={r.get('seed','')} F1={r.get('strict_f1','')}")

# ============================================================
# Generic train+eval for AG News (P14) and SALAD (P15)
# ============================================================
def train_and_eval(dataset_name, peft_type, seed, paper,
                   train_size=5000, test_size=1000,
                   epochs=3, lr=2e-4, batch=4, gacc=8,
                   rank=64, max_len=512):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"📊 {paper} | {peft_type} | {dataset_name} | seed={seed}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"

    mdl = AutoModelForCausalLM.from_pretrained(MODEL,
        dtype=torch.float16, device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()

    # PEFT config
    if peft_type == "lora":
        cfg = LoraConfig(r=rank, lora_alpha=rank*2,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=0, bias="none", task_type="CAUSAL_LM")
    elif peft_type == "oft":
        cfg = OFTConfig(r=rank,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            module_dropout=0)
    mdl = get_peft_model(mdl, cfg)
    params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Trainable params: {params:,}")

    # Load dataset
    if dataset_name == "ag_news":
        ds_raw = load_dataset("ag_news", split="train").shuffle(seed=seed).select(range(train_size))
        test_raw = load_dataset("ag_news", split="test").shuffle(seed=seed).select(range(test_size))
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        def make_msg(ex):
            return [{"role":"user","content":f"Classify this news: {ex['text'][:300]}"},
                    {"role":"assistant","content":labels[ex['label']]}]
        train_msgs = [make_msg(ex) for ex in ds_raw]
        test_msgs = [(make_msg(ex), labels[ex['label']]) for ex in test_raw]
    elif dataset_name == "salad_multitask":
        salad_train = json.load(open("/workspace/salad_data/train_5k_clean.json"))
        salad_test = json.load(open("/workspace/salad_data/test_held_out.json"))[:test_size]
        train_msgs = []
        for d in salad_train:
            c = d["conversations"]
            train_msgs.append([{"role":"system","content":c[0]["value"]},
                              {"role":"user","content":c[1]["value"]},
                              {"role":"assistant","content":c[2]["value"]}])
        test_msgs = []
        for d in salad_test:
            c = d["conversations"]
            test_msgs.append(([{"role":"system","content":c[0]["value"]},
                              {"role":"user","content":c[1]["value"]},
                              {"role":"assistant","content":c[2]["value"]}],
                             c[2]["value"]))

    # Tokenize
    def tokenize_batch(msgs_list):
        all_ids, all_attn, all_labels = [], [], []
        for msgs in msgs_list:
            s = tok.apply_chat_template(msgs, tokenize=False)
            enc = tok(s, truncation=True, max_length=max_len, padding="max_length")
            ids = enc["input_ids"]; attn = enc["attention_mask"]
            lb = [l if a==1 else -100 for l,a in zip(ids, attn)]
            all_ids.append(ids); all_attn.append(attn); all_labels.append(lb)
        return Dataset.from_dict({"input_ids":all_ids,"attention_mask":all_attn,"labels":all_labels})

    ds = tokenize_batch(train_msgs)
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 10, (len(loader)//gacc)*epochs)

    mdl.train(); t0 = time.time()
    for ep in range(epochs):
        tl=0; opt.zero_grad()
        for st, b in enumerate(loader):
            b = {k:v.to("cuda") for k,v in b.items()}
            loss = mdl(**b).loss/gacc; loss.backward(); tl+=loss.item()*gacc
            if (st+1)%gacc==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"  Epoch {ep+1}/{epochs} loss={tl/len(loader):.4f}")
    train_min = (time.time()-t0)/60

    # Eval
    mdl.eval(); preds, golds = [], []
    for msgs, gold in test_msgs:
        prompt_msgs = msgs[:-1]  # remove assistant
        p = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt", truncation=True, max_length=max_len).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=100, do_sample=False)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append(resp.split("\n")[0].strip())
        golds.append(gold if dataset_name=="ag_news" else gold.split("\n")[0].strip())

    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    acc = sum(p==g for p,g in zip(preds,golds))/len(golds)
    print(f"  F1={f1:.4f} Acc={acc:.4f} ({train_min:.1f}m)")

    result = {
        "paper":paper, "peft_type":peft_type, "dataset":dataset_name,
        "seed":seed, "strict_f1":round(f1,4), "accuracy":round(acc,4),
        "train_min":round(train_min,1), "trainable_params":params,
        "gpu":torch.cuda.get_device_name(0), "ts":datetime.now().isoformat()
    }
    autosave(result, f"{paper}_{peft_type}_{dataset_name}_s{seed}.json")
    del mdl, tok; torch.cuda.empty_cache()
    return result

# ============================================================
# Run experiments
# ============================================================
ALL = []

# P14: LoRA vs OFT on AG News — 2 new seeds
print("\n" + "🔵"*30)
print("P14: LoRA vs OFT — Seeds 456, 999")
for seed in [456, 999]:
    ALL.append(train_and_eval("ag_news", "lora", seed, "P14"))
    ALL.append(train_and_eval("ag_news", "oft", seed, "P14"))

# P15: Multi-task SALAD — 2 new seeds
print("\n" + "🟢"*30)
print("P15: Multi-task SALAD — Seeds 456, 999")
for seed in [456, 999]:
    ALL.append(train_and_eval("salad_multitask", "lora", seed, "P15"))

# Summary
print("\n" + "="*70)
print("📊 SEED EXPANSION RESULTS")
print("="*70)
for r in ALL:
    print(f"  {r['paper']:<5} {r['peft_type']:<5} {r['dataset']:<16} s={r['seed']:<4} F1={r['strict_f1']}")
print(f"\nAll saved: {SAVE}/")
PYEOF

echo "🏁 Seed expansion done — $(date)"

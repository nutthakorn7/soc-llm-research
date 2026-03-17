#!/bin/bash
# P8 LedGAR + P6 Seeds — Vast.ai
# ================================
# P8 LedGAR: High-class (100 classes) legal docs — 3 seeds
# P6: SALAD scaling with 0.5B (2 new seeds 456, 999) — placeholder for 9B
# Runs after P14/P15 seeds on 4090

set -e
echo "🚀 P8 LedGAR + P6 0.5B seeds — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

pip install -q transformers>=4.45 peft>=0.13 accelerate>=0.34 datasets scikit-learn 2>/dev/null

mkdir -p /workspace/results/p8_ledgar
mkdir -p /workspace/results/p6_seeds

python3 << 'PYEOF'
import torch, random, json, time, os, csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import Dataset, load_dataset

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def autosave(r, fn, save_dir):
    with open(f"{save_dir}/{fn}","w") as f: json.dump(r,f,indent=2)
    cp = f"{save_dir}/summary.csv"; ex = os.path.exists(cp)
    with open(cp,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        if not ex: w.writeheader()
        w.writerow(r)
    print(f"✅ SAVED {fn} | F1={r.get('strict_f1','?')}")

def train_eval(dataset_name, seed, paper, save_dir,
               train_size=5000, test_size=1000,
               epochs=3, lr=2e-4, batch=4, gacc=8, rank=64, ml=512):
    random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*60}\n📊 {paper} | {dataset_name} | seed={seed}\n{'='*60}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    mdl = AutoModelForCausalLM.from_pretrained(MODEL,
        dtype=torch.float16, device_map="auto", trust_remote_code=True)
    mdl.enable_input_require_grads()
    mdl = get_peft_model(mdl, LoraConfig(r=rank, lora_alpha=rank*2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))

    # Load dataset
    if dataset_name == "ledgar":
        ds_raw = load_dataset("lex_glue", "ledgar", split="train").shuffle(seed=seed)
        test_raw = load_dataset("lex_glue", "ledgar", split="test").shuffle(seed=seed)
        label_names = ds_raw.features["label"].names
        n_classes = len(label_names)
        print(f"  LedGAR: {n_classes} classes, H={__import__('math').log2(n_classes):.2f} bits")
        ds_raw = ds_raw.select(range(min(train_size, len(ds_raw))))
        test_raw = test_raw.select(range(min(test_size, len(test_raw))))
        def make_msgs(ex):
            return ([{"role":"user","content":f"Classify this legal provision: {ex['text'][:400]}"},
                     {"role":"assistant","content":label_names[ex['label']]}],
                    label_names[ex['label']])
    elif dataset_name == "salad":
        salad = json.load(open("/workspace/salad_data/train_5k_clean.json"))
        salad_test = json.load(open("/workspace/salad_data/test_held_out.json"))[:test_size]
        random.shuffle(salad)
        ds_items = salad[:train_size]
        def make_salad_msgs(d):
            c = d["conversations"]
            return ([{"role":"system","content":c[0]["value"]},
                     {"role":"user","content":c[1]["value"]},
                     {"role":"assistant","content":c[2]["value"]}],
                    c[2]["value"].split("\n")[0].strip())
        ds_raw = ds_items; test_raw = salad_test
        make_msgs = make_salad_msgs

    # Tokenize train
    train_msgs = [make_msgs(ex) for ex in ds_raw] if dataset_name != "salad" else [make_salad_msgs(ex) for ex in ds_items]
    test_msgs = [make_msgs(ex) for ex in test_raw] if dataset_name != "salad" else [make_salad_msgs(ex) for ex in salad_test]

    all_ids, all_attn, all_labels = [], [], []
    for msgs, _ in train_msgs:
        s = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(s, truncation=True, max_length=ml, padding="max_length")
        ids=enc["input_ids"]; attn=enc["attention_mask"]
        lb = [l if a==1 else -100 for l,a in zip(ids,attn)]
        all_ids.append(ids); all_attn.append(attn); all_labels.append(lb)
    ds = Dataset.from_dict({"input_ids":all_ids,"attention_mask":all_attn,"labels":all_labels})
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 10, (len(loader)//gacc)*epochs)

    mdl.train(); t0=time.time()
    for ep in range(epochs):
        tl=0; opt.zero_grad()
        for st,b in enumerate(loader):
            b={k:v.to("cuda") for k,v in b.items()}
            loss=mdl(**b).loss/gacc; loss.backward(); tl+=loss.item()*gacc
            if (st+1)%gacc==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"  Epoch {ep+1}/{epochs} loss={tl/len(loader):.4f}")
    tm=(time.time()-t0)/60

    # Eval
    mdl.eval(); preds, golds = [], []
    for msgs, gold in test_msgs[:test_size]:
        prompt_msgs = msgs[:-1]
        p = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt", truncation=True, max_length=ml).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=50, do_sample=False)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append(resp.split("\n")[0].strip()); golds.append(gold)
        if len(preds)%200==0:
            print(f"  eval {len(preds)}/{test_size} f1={f1_score(golds,preds,average='macro',zero_division=0):.4f}")

    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    acc = sum(p==g for p,g in zip(preds,golds))/len(golds)
    result = {"paper":paper,"dataset":dataset_name,"seed":seed,
              "strict_f1":round(f1,4),"accuracy":round(acc,4),
              "train_min":round(tm,1),"n_classes":len(set(golds)),
              "gpu":torch.cuda.get_device_name(0),"ts":datetime.now().isoformat()}
    autosave(result, f"{paper}_{dataset_name}_s{seed}.json", save_dir)
    del mdl, tok; torch.cuda.empty_cache()
    return result

ALL = []

# === P8 LedGAR: 3 seeds ===
print("\n" + "🔴"*30 + "\nP8 LedGAR (100 classes)")
for seed in [42, 77, 123]:
    ALL.append(train_eval("ledgar", seed, "P8", "/workspace/results/p8_ledgar"))

# === P6: SALAD 0.5B seeds (as proxy for 9B trends) ===
print("\n" + "🟡"*30 + "\nP6 SALAD 0.5B (seeds 456, 999)")
for seed in [456, 999]:
    ALL.append(train_eval("salad", seed, "P6_0.5B", "/workspace/results/p6_seeds"))

print("\n" + "="*70 + "\n📊 ALL RESULTS\n" + "="*70)
for r in ALL:
    print(f"  {r['paper']:<8} {r['dataset']:<10} s={r['seed']:<4} F1={r['strict_f1']} classes={r.get('n_classes','?')}")
PYEOF

echo "🏁 P8 LedGAR + P6 seeds done — $(date)"

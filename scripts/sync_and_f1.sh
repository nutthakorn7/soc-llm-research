#!/bin/bash
# sync_and_f1.sh — One command to sync all eval results + calculate F1
# Usage: bash scripts/sync_and_f1.sh

set -e
REMOTE="lanta:/project/lt200473-ttctvs/soc-finetune"
LOCAL="/Users/pop7/Code/Lanta/results"

echo "======================================================================="
echo "  SOC-FT: Sync + F1 Calculator"
echo "  $(date)"
echo "======================================================================="

# 1. Check which evals are complete on Lanta
echo ""
echo "📡 Checking Lanta for completed evals..."
COMPLETED=$(ssh lanta 'for d in /project/lt200473-ttctvs/soc-finetune/outputs/eval-*/generated_predictions.jsonl; do basename $(dirname "$d"); done 2>/dev/null')

if [ -z "$COMPLETED" ]; then
    echo "  ❌ No completed evals found!"
    exit 1
fi

COUNT=$(echo "$COMPLETED" | wc -l | tr -d ' ')
echo "  Found $COUNT completed evals"

# 2. Sync predictions
echo ""
echo "📥 Syncing predictions..."
for eval_name in $COMPLETED; do
    mkdir -p "$LOCAL/$eval_name"
    echo "  Syncing $eval_name..."
    rsync -az "$REMOTE/outputs/$eval_name/generated_predictions.jsonl" "$LOCAL/$eval_name/" 2>/dev/null || true
done

# 3. Sync test data if needed
if [ ! -f "$LOCAL/test_held_out.json" ]; then
    echo "  Syncing test_held_out.json..."
    rsync -az "$REMOTE/data/test_held_out.json" "$LOCAL/" 2>/dev/null || true
fi

# 4. Calculate F1 for each eval
echo ""
echo "🧮 Calculating F1 scores..."
echo ""

python3 - << 'PYTHON_SCRIPT'
import json, os, sys
from collections import Counter

RESULTS_DIR = "/Users/pop7/Code/Lanta/results"
TEST_FILE = os.path.join(RESULTS_DIR, "test_held_out.json")

# Load ground truth
with open(TEST_FILE) as f:
    test_data = json.load(f)

def parse_labels(text):
    import re
    # Strip <think>...</think> tags (Qwen3.5 thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    labels = {}
    for line in text.split("\n"):
        if "Classification:" in line: labels["cls"] = line.split("Classification:")[1].strip()
        elif "Triage:" in line: labels["tri"] = line.split("Triage:")[1].strip()
        elif "Attack Category:" in line: labels["atk"] = line.split("Attack Category:")[1].strip()
        elif "Priority Score:" in line:
            try: labels["pri"] = float(line.split("Priority Score:")[1].strip())
            except: pass
    return labels

# Ground truth
gt_labels = []
for item in test_data:
    conv = item["conversations"]
    asst = conv[2]["value"] if len(conv) > 2 else ""
    gt_labels.append(parse_labels(asst))

# Fuzzy match for known variations
FUZZY = {
    "backdoors": "Backdoor", "backdoor": "Backdoor",
    "dos": "DoS", "denial of service": "DoS", "ddos": "DoS",
    "exploits": "Exploits", "exploit": "Exploits",
    "reconnaissance": "Reconnaissance", "recon": "Reconnaissance",
    "fuzzers": "Fuzzers", "fuzzer": "Fuzzers",
    "generic": "Generic", "analysis": "Analysis", "benign": "Benign",
    "malicious": "Malicious", "brute force": "Brute Force",
}

def normalize(val):
    if not val: return val
    lower = val.lower().strip()
    return FUZZY.get(lower, val)

def calc_f1(true_list, pred_list):
    """Macro F1"""
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for t, p in zip(true_list, pred_list):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    
    f1s = []
    for cls in set(list(tp.keys()) + list(fn.keys())):
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        f1s.append(f1)
    return sum(f1s)/len(f1s)*100 if f1s else 0

# Process each eval
results = []
eval_dirs = sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("eval-") and os.path.isdir(os.path.join(RESULTS_DIR, d))])

print(f"{'Model':<30} {'Cls':>6} {'Tri':>6} {'Atk':>6} {'Atk*':>6} {'PriMAE':>8} {'Avg':>6}")
print("-" * 80)

for eval_name in eval_dirs:
    pred_file = os.path.join(RESULTS_DIR, eval_name, "generated_predictions.jsonl")
    if not os.path.exists(pred_file):
        continue
    
    with open(pred_file) as f:
        preds = [json.loads(line) for line in f]
    
    pred_labels = [parse_labels(p.get("predict", "")) for p in preds]
    
    n = min(len(gt_labels), len(pred_labels))
    
    # Strict F1
    cls_true = [g.get("cls","?") for g in gt_labels[:n]]
    cls_pred = [p.get("cls","?") for p in pred_labels[:n]]
    tri_true = [g.get("tri","?") for g in gt_labels[:n]]
    tri_pred = [p.get("tri","?") for p in pred_labels[:n]]
    atk_true = [g.get("atk","?") for g in gt_labels[:n]]
    atk_pred = [p.get("atk","?") for p in pred_labels[:n]]
    
    f1_cls = calc_f1(cls_true, cls_pred)
    f1_tri = calc_f1(tri_true, tri_pred)
    f1_atk = calc_f1(atk_true, atk_pred)
    
    # Fuzzy F1 (normalize attack category)
    atk_pred_fuzzy = [normalize(p.get("atk","?")) for p in pred_labels[:n]]
    atk_true_norm = [normalize(g.get("atk","?")) for g in gt_labels[:n]]
    f1_atk_fuzzy = calc_f1(atk_true_norm, atk_pred_fuzzy)
    
    # Priority MAE
    pri_errors = []
    for g, p in zip(gt_labels[:n], pred_labels[:n]):
        if "pri" in g and "pri" in p:
            pri_errors.append(abs(g["pri"] - p["pri"]))
    mae = sum(pri_errors)/len(pri_errors) if pri_errors else -1
    
    avg = (f1_cls + f1_tri + f1_atk_fuzzy) / 3
    mae_str = f"{mae:.4f}" if mae >= 0 else "N/A"
    
    # Flag if different with fuzzy
    atk_marker = f"{f1_atk:.1f}" if abs(f1_atk - f1_atk_fuzzy) < 0.01 else f"{f1_atk:.1f}"
    
    print(f"  {eval_name:<28} {f1_cls:>5.1f}% {f1_tri:>5.1f}% {f1_atk:>5.1f}% {f1_atk_fuzzy:>5.1f}% {mae_str:>8} {avg:>5.1f}%")
    
    results.append({
        "eval": eval_name,
        "n_samples": n,
        "f1_classification": round(f1_cls, 2),
        "f1_triage": round(f1_tri, 2),
        "f1_attack_category": round(f1_atk, 2),
        "f1_attack_category_fuzzy": round(f1_atk_fuzzy, 2),
        "priority_mae": round(mae, 4) if mae >= 0 else None,
        "avg_f1_fuzzy": round(avg, 2),
    })

print("-" * 80)
print(f"  * Atk* = Fuzzy match (normalizes Backdoors→Backdoor, etc.)")

# Save
out_file = os.path.join(RESULTS_DIR, "paper_results", "all_f1_results.json")
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved to {out_file}")
PYTHON_SCRIPT

echo ""
echo "======================================================================="
echo "  Done! Results at: $LOCAL/paper_results/all_f1_results.json"
echo "======================================================================="

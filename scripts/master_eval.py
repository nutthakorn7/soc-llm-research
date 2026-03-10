#!/usr/bin/env python3
"""
SOC-FT Master Auto-Evaluation Pipeline
Run this AFTER training jobs complete. It will:
1. Run F1 evaluation on all predictions
2. Run comprehensive baselines
3. Run leave-one-out generalization test
4. Run sanity checks
5. Generate paper-ready results table
6. Save everything to outputs/paper_results/
"""
import json, os, sys, re, glob, math
import numpy as np
from collections import Counter, defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

BASE = sys.argv[1] if len(sys.argv) > 1 else "/project/lt200473-ttctvs/soc-finetune"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "outputs")
PAPER = os.path.join(OUT, "paper_results")
os.makedirs(PAPER, exist_ok=True)

FEAT_NAMES = ["Alert Type", "Severity", "Protocol", "MITRE Tactic",
              "MITRE Technique", "Kill Chain Phase", "Network Segment"]

BASELINES = {
    "Claude Opus 4.6 (5S)":   {"f1": 0.907, "cost": 672},
    "Claude Opus 4.6 (AVG)":  {"f1": 0.815, "cost": 672},
    "Kimi K2 (AVG)":          {"f1": 0.729, "cost": 0},
    "Llama 4 Maverick (AVG)": {"f1": 0.721, "cost": 7},
    "DeepSeek V3.2 (AVG)":    {"f1": 0.709, "cost": 9},
    "Grok 4.1 Fast (AVG)":    {"f1": 0.603, "cost": 5},
    "GPT-5.2 (AVG)":          {"f1": 0.602, "cost": 75},
}

# ============================================================
# 1. PARSE HELPERS
# ============================================================
def parse_sample(conv):
    if len(conv) < 3: return None, None
    feats = {}
    for l in conv[1]["value"].split("\n"):
        if ":" in l:
            k, v = l.split(":", 1)
            feats[k.strip()] = v.strip()
    labels = {}
    resp = re.sub(r'<think>.*?</think>', '', conv[2]["value"], flags=re.DOTALL).strip()
    for l in resp.split("\n"):
        ll = l.strip().lower()
        if ll.startswith("classification:"): labels["cls"] = ll.split(":", 1)[1].strip()
        elif "triage" in ll and ":" in ll: labels["tri"] = ll.split(":", 1)[1].strip()
        elif ll.startswith("attack category:"): labels["atk"] = ll.split(":", 1)[1].strip()
        elif ll.startswith("priority score:"):
            try: labels["pri"] = float(ll.split(":", 1)[1].strip())
            except: pass
    return feats, labels

def parse_prediction(text):
    result = {"cls": None, "tri": None, "atk": None, "pri": None}
    if not text: return result
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    for l in text.split("\n"):
        ll = l.strip().lower()
        if ll.startswith("classification:"): result["cls"] = ll.split(":", 1)[1].strip()
        elif "triage" in ll and ":" in ll: result["tri"] = ll.split(":", 1)[1].strip()
        elif ll.startswith("attack category:"): result["atk"] = ll.split(":", 1)[1].strip()
        elif ll.startswith("priority score:"):
            try: result["pri"] = float(ll.split(":", 1)[1].strip())
            except: pass
    return result

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

# ============================================================
# 2. EVALUATE LLM PREDICTIONS
# ============================================================
def eval_llm_predictions(pred_file, name="model"):
    """Evaluate a single prediction file."""
    with open(pred_file) as f:
        data = [json.loads(l) for l in f]
    
    y = {"cls": ([], []), "tri": ([], []), "atk": ([], [])}
    pri_errors = []
    exact = 0
    
    for item in data:
        label = parse_prediction(item.get("label", ""))
        pred = parse_prediction(item.get("predict", ""))
        for task in ["cls", "tri", "atk"]:
            if label[task] and pred[task]:
                y[task][0].append(label[task])
                y[task][1].append(pred[task])
        if label["pri"] is not None and pred["pri"] is not None:
            pri_errors.append(abs(label["pri"] - pred["pri"]))
        if label["cls"] == pred["cls"] and label["tri"] == pred["tri"] and label["atk"] == pred["atk"]:
            exact += 1
    
    results = {"name": name, "n_samples": len(data)}
    for task in ["cls", "tri", "atk"]:
        if y[task][0]:
            results[task + "_f1"] = macro_f1(y[task][0], y[task][1])
            results[task + "_n"] = len(y[task][0])
        else:
            results[task + "_f1"] = None
    
    task_f1s = [results.get(t + "_f1", 0) for t in ["cls", "tri", "atk"] if results.get(t + "_f1") is not None]
    results["avg_f1"] = sum(task_f1s) / len(task_f1s) if task_f1s else 0
    results["exact_match"] = exact / len(data) if data else 0
    results["priority_mae"] = sum(pri_errors) / len(pri_errors) if pri_errors else None
    
    # Confusion matrix for attack category
    if y["atk"][0]:
        labels_sorted = sorted(set(y["atk"][0] + y["atk"][1]))
        cm = confusion_matrix(y["atk"][0], y["atk"][1], labels=labels_sorted)
        results["atk_confusion"] = {
            "labels": labels_sorted,
            "matrix": cm.tolist()
        }
        results["atk_classification_report"] = classification_report(
            y["atk"][0], y["atk"][1], output_dict=True, zero_division=0
        )
    
    return results

# ============================================================
# 3. COMPREHENSIVE BASELINES
# ============================================================
def run_all_baselines():
    """Run ALL traditional ML baselines."""
    with open(os.path.join(DATA, "train_5k_clean.json")) as f: train = json.load(f)
    with open(os.path.join(DATA, "test_held_out.json")) as f: test = json.load(f)
    
    encs = {f: LabelEncoder() for f in FEAT_NAMES}
    all_vals = {f: set() for f in FEAT_NAMES}
    for d in train + test:
        ft, _ = parse_sample(d["conversations"])
        if ft:
            for f in FEAT_NAMES: all_vals[f].add(ft.get(f, "unk"))
    for f in FEAT_NAMES: encs[f].fit(list(all_vals[f]) + ["unk"])
    
    def encode(data):
        X, yc, yt, ya = [], [], [], []
        for d in data:
            ft, lb = parse_sample(d["conversations"])
            if not ft or not lb or "cls" not in lb: continue
            row = [encs[f].transform([ft.get(f,"unk") if ft.get(f,"unk") in encs[f].classes_ else "unk"])[0] for f in FEAT_NAMES]
            X.append(row); yc.append(lb.get("cls","unk")); yt.append(lb.get("tri","unk")); ya.append(lb.get("atk","unk"))
        return np.array(X), yc, yt, ya
    
    Xtr, yc_tr, yt_tr, ya_tr = encode(train)
    Xte, yc_te, yt_te, ya_te = encode(test)
    
    tasks = [("cls", yc_tr, yc_te), ("tri", yt_tr, yt_te), ("atk", ya_tr, ya_te)]
    results = {}
    
    models = [
        ("Random", None),
        ("Majority Vote", None),
        ("DT (depth=5)", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("DT (unlimited)", DecisionTreeClassifier(random_state=42)),
        ("RF (n=100)", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("GBM (n=100)", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
    
    for model_name, model in models:
        scores = {}
        for task, ytr, yte in tasks:
            le = LabelEncoder(); ytr_e = le.fit_transform(ytr)
            if model_name == "Random":
                np.random.seed(42)
                yp = [np.random.choice(list(set(ytr))) for _ in yte]
            elif model_name == "Majority Vote":
                maj = Counter(ytr).most_common(1)[0][0]
                yp = [maj] * len(yte)
            else:
                m = type(model)(**model.get_params())
                m.fit(Xtr, ytr_e)
                yp = le.inverse_transform(m.predict(Xte))
            scores[task + "_f1"] = macro_f1(yte, yp)
        
        scores["avg_f1"] = sum(scores[t+"_f1"] for t in ["cls","tri","atk"]) / 3
        results[model_name] = scores
    
    return results

# ============================================================
# 4. LEAVE-ONE-OUT GENERALIZATION
# ============================================================
def run_generalization():
    """Hold out entire attack types. Test if DT and LLM generalize."""
    with open(os.path.join(DATA, "train_5k_clean.json")) as f: train = json.load(f)
    
    # Group by alert_type
    by_type = defaultdict(list)
    for d in train:
        ft, _ = parse_sample(d["conversations"])
        if ft: by_type[ft.get("Alert Type", "unk")].append(d)
    
    results = {}
    encs = {f: LabelEncoder() for f in FEAT_NAMES}
    
    for holdout_type, holdout_data in sorted(by_type.items()):
        if len(holdout_data) < 20: continue
        remaining = [d for t, ds in by_type.items() if t != holdout_type for d in ds]
        
        # DT baseline on remaining → predict holdout
        all_vals = {f: set() for f in FEAT_NAMES}
        for d in remaining + holdout_data:
            ft, _ = parse_sample(d["conversations"])
            if ft:
                for f in FEAT_NAMES: all_vals[f].add(ft.get(f, "unk"))
        encs_local = {f: LabelEncoder() for f in FEAT_NAMES}
        for f in FEAT_NAMES: encs_local[f].fit(list(all_vals[f]) + ["unk"])
        
        def enc(data):
            X, ya = [], []
            for d in data:
                ft, lb = parse_sample(d["conversations"])
                if not ft or not lb: continue
                row = [encs_local[f].transform([ft.get(f,"unk") if ft.get(f,"unk") in encs_local[f].classes_ else "unk"])[0] for f in FEAT_NAMES]
                X.append(row); ya.append(lb.get("atk", "unk"))
            return np.array(X) if X else np.array([]).reshape(0, len(FEAT_NAMES)), ya
        
        Xtr, ya_tr = enc(remaining)
        Xte, ya_te = enc(holdout_data)
        
        if len(Xtr) < 10 or len(Xte) < 5: continue
        
        le = LabelEncoder()
        ya_tr_e = le.fit_transform(ya_tr)
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(Xtr, ya_tr_e)
        yp = le.inverse_transform(dt.predict(Xte))
        dt_f1 = macro_f1(ya_te, yp)
        
        results[holdout_type] = {
            "train_size": len(remaining),
            "test_size": len(holdout_data),
            "dt_f1": dt_f1
        }
    
    return results

# ============================================================
# 5. TASK COMPLEXITY
# ============================================================
def task_complexity():
    with open(os.path.join(DATA, "train_5k_clean.json")) as f: data = json.load(f)
    labels = {"cls": [], "tri": [], "atk": []}
    for d in data:
        _, lb = parse_sample(d["conversations"])
        if lb:
            for t in ["cls", "tri", "atk"]:
                if t in lb: labels[t].append(lb[t])
    
    results = {}
    for task, vals in labels.items():
        c = Counter(vals)
        total = sum(c.values())
        h = -sum((v/total) * math.log2(v/total) for v in c.values() if v > 0)
        nc = len(c)
        mh = math.log2(nc) if nc > 1 else 0
        results[task] = {
            "n_classes": nc, "entropy": round(h, 3),
            "max_entropy": round(mh, 3),
            "norm_entropy": round(h/mh if mh else 0, 3),
            "distribution": dict(c)
        }
    return results

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  SOC-FT MASTER EVALUATION PIPELINE")
    print("=" * 70)
    
    all_results = {}
    
    # 1. Task complexity
    print("\n📐 Task Complexity Analysis...")
    tc = task_complexity()
    all_results["task_complexity"] = tc
    for task, r in tc.items():
        print("  %s: %d classes, H=%.3f (norm=%.3f)" % (task, r["n_classes"], r["entropy"], r["norm_entropy"]))
    
    # 2. Baselines
    print("\n📊 Traditional ML Baselines...")
    baselines = run_all_baselines()
    all_results["baselines"] = baselines
    print("  %-20s %8s %8s %8s %8s" % ("Method", "Cls", "Tri", "Atk", "Avg"))
    print("  " + "-" * 56)
    for name, r in baselines.items():
        print("  %-20s %8.4f %8.4f %8.4f %8.4f" % (name, r["cls_f1"], r["tri_f1"], r["atk_f1"], r["avg_f1"]))
    
    # 3. LLM evaluations
    print("\n🤖 LLM Evaluations...")
    llm_results = {}
    for eval_dir in sorted(glob.glob(os.path.join(OUT, "eval-*"))):
        pred_file = os.path.join(eval_dir, "generated_predictions.jsonl")
        if os.path.exists(pred_file):
            name = os.path.basename(eval_dir).replace("eval-", "")
            r = eval_llm_predictions(pred_file, name)
            llm_results[name] = r
            print("  %-30s Cls=%.4f Tri=%.4f Atk=%.4f Avg=%.4f" % (
                name, r.get("cls_f1",0) or 0, r.get("tri_f1",0) or 0, r.get("atk_f1",0) or 0, r["avg_f1"]))
    all_results["llm_results"] = llm_results
    
    # 4. Generalization test
    print("\n🧪 Generalization (Leave-One-Out by Alert Type)...")
    gen = run_generalization()
    all_results["generalization"] = gen
    for atype, r in gen.items():
        print("  Hold out %-25s: DT F1=%.4f (train=%d, test=%d)" % (atype, r["dt_f1"], r["train_size"], r["test_size"]))
    
    # 5. Paper comparison table
    print("\n" + "=" * 70)
    print("  📋 PAPER-READY COMPARISON TABLE")
    print("=" * 70)
    print("  %-30s %-10s %8s %8s %8s %8s" % ("Model", "Method", "Cls", "Tri", "Atk", "Avg"))
    print("  " + "-" * 74)
    
    # Our models
    for name, r in llm_results.items():
        print("  %-30s %-10s %8.4f %8.4f %8.4f %8.4f" % (
            name, "FT", r.get("cls_f1",0) or 0, r.get("tri_f1",0) or 0, r.get("atk_f1",0) or 0, r["avg_f1"]))
    print("  " + "-" * 74)
    
    # Traditional baselines
    for name, r in baselines.items():
        print("  %-30s %-10s %8.4f %8.4f %8.4f %8.4f" % (name, "ML", r["cls_f1"], r["tri_f1"], r["atk_f1"], r["avg_f1"]))
    print("  " + "-" * 74)
    
    # ICL baselines
    for name, b in BASELINES.items():
        print("  %-30s %-10s %8s %8s %8s %8.4f" % (name, "ICL", "—", "—", "—", b["f1"]))
    
    # Save
    with open(os.path.join(PAPER, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n  Saved to %s/all_results.json" % PAPER)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
P18: Zero-Shot Transfer — Leave-One-Category-Out
1. Creates 8 data splits (each excluding 1 attack category)
2. Runs DT baseline (immediate results)
3. Generates LLM training scripts for Lanta
"""
import json, os, sys
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report

DATA_DIR = "/Users/pop7/Code/Lanta/results"
RESULTS_DIR = "/Users/pop7/Code/Lanta/results/paper_results"

FEAT_NAMES = ["Alert Type", "Severity", "Protocol", "MITRE Tactic",
              "MITRE Technique", "Kill Chain Phase", "Network Segment"]

def parse_sample(item):
    conv = item["conversations"]
    user_text = conv[1]["value"] if len(conv) > 1 else ""
    asst_text = conv[2]["value"] if len(conv) > 2 else ""
    feats, labels = {}, {}
    for line in user_text.split("\n"):
        for fn in FEAT_NAMES:
            if f"{fn}:" in line:
                feats[fn] = line.split(f"{fn}:")[1].strip()
    for line in asst_text.split("\n"):
        if "Classification:" in line: labels["cls"] = line.split("Classification:")[1].strip()
        elif "Triage:" in line: labels["tri"] = line.split("Triage:")[1].strip()
        elif "Attack Category:" in line: labels["atk"] = line.split("Attack Category:")[1].strip()
    return feats, labels

def encode_features(data, encoders=None):
    if encoders is None:
        encoders = {}
        for fn in FEAT_NAMES:
            enc = LabelEncoder()
            vals = [d.get(fn, "unknown") for d in data]
            enc.fit(list(set(vals) | {"unknown"}))
            encoders[fn] = enc
    X = np.zeros((len(data), len(FEAT_NAMES)))
    for i, d in enumerate(data):
        for j, fn in enumerate(FEAT_NAMES):
            val = d.get(fn, "unknown")
            if val in encoders[fn].classes_:
                X[i, j] = encoders[fn].transform([val])[0]
            else:
                X[i, j] = -1
    return X, encoders

def main():
    print("=" * 70)
    print("  P18: Zero-Shot Transfer — Leave-One-Category-Out")
    print("=" * 70)

    # Load test data (use as both source for splits)
    test_file = os.path.join(DATA_DIR, "test_held_out.json")
    if not os.path.exists(test_file):
        # Try syncing
        print(f"  ❌ {test_file} not found!")
        sys.exit(1)

    with open(test_file) as f:
        test_data = json.load(f)

    # Also load train data if available
    train_file = os.path.join(DATA_DIR, "outputs/clean-qwen35-5k/trainer_log.jsonl")
    
    # Parse all samples
    all_feats, all_labels = [], []
    for item in test_data:
        f, l = parse_sample(item)
        if f and l and "atk" in l:
            all_feats.append(f)
            all_labels.append(l)

    # Get unique categories
    categories = sorted(set(l["atk"] for l in all_labels))
    cat_counts = Counter(l["atk"] for l in all_labels)
    
    print(f"\n  Total samples: {len(all_feats)}")
    print(f"  Categories ({len(categories)}):")
    for cat in categories:
        print(f"    {cat}: {cat_counts[cat]} samples")

    # --- DT Leave-One-Out Experiment ---
    print(f"\n{'='*70}")
    print(f"  DT Baseline: Leave-One-Category-Out")
    print(f"{'='*70}")

    dt_results = []
    for held_out_cat in categories:
        # Split: train on everything EXCEPT held_out_cat
        train_idx = [i for i, l in enumerate(all_labels) if l["atk"] != held_out_cat]
        test_idx = [i for i, l in enumerate(all_labels) if l["atk"] == held_out_cat]

        train_f = [all_feats[i] for i in train_idx]
        train_l = [all_labels[i] for i in train_idx]
        test_f = [all_feats[i] for i in test_idx]
        test_l = [all_labels[i] for i in test_idx]

        # Encode
        X_train, encoders = encode_features(train_f)
        X_test, _ = encode_features(test_f, encoders)

        # Train DT for attack category
        le = LabelEncoder()
        # Include held_out_cat in encoder so inverse_transform works
        all_cats = list(set(l["atk"] for l in train_l) | {held_out_cat})
        le.fit(all_cats)
        y_train = le.transform([l["atk"] for l in train_l])
        y_test = le.transform([l["atk"] for l in test_l])

        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        # Accuracy (not F1 since it's single class)
        acc = (y_pred == y_test).mean()
        pred_labels = le.inverse_transform(y_pred)
        pred_dist = Counter(pred_labels)

        result = {
            "held_out": held_out_cat,
            "n_test": len(test_idx),
            "n_train": len(train_idx),
            "dt_accuracy": round(acc * 100, 1),
            "predicted_as": dict(pred_dist.most_common(3)),
        }
        dt_results.append(result)

    # Print DT results
    print(f"\n  {'Held-Out Category':<20} {'N Test':>7} {'DT Acc':>8} {'Predicted As'}")
    print(f"  {'-'*65}")
    for r in dt_results:
        preds_str = ", ".join(f"{k}({v})" for k, v in r["predicted_as"].items())
        marker = " ✅" if r["dt_accuracy"] > 50 else " ❌"
        print(f"  {r['held_out']:<20} {r['n_test']:>7} {r['dt_accuracy']:>7.1f}%{marker} {preds_str}")

    avg_acc = np.mean([r["dt_accuracy"] for r in dt_results])
    print(f"\n  Average DT Zero-Shot Accuracy: {avg_acc:.1f}%")

    # --- Generate Lanta training scripts for 0.8B ---
    print(f"\n{'='*70}")
    print(f"  LLM Training Scripts (Qwen3.5-0.8B)")
    print(f"{'='*70}")

    lanta_base = "/project/lt200473-ttctvs/soc-finetune"
    splits_dir = os.path.join(RESULTS_DIR, "zero_shot_splits")
    os.makedirs(splits_dir, exist_ok=True)

    # Save data splits for Lanta
    for held_out_cat in categories:
        safe_name = held_out_cat.lower().replace(" ", "_")
        
        # Train split (exclude held_out_cat)
        train_items = [item for item, (f, l) in zip(test_data, zip(all_feats, all_labels)) 
                       if l.get("atk") != held_out_cat]
        test_items = [item for item, (f, l) in zip(test_data, zip(all_feats, all_labels)) 
                      if l.get("atk") == held_out_cat]
        
        with open(os.path.join(splits_dir, f"train_no_{safe_name}.json"), "w") as f:
            json.dump(train_items, f)
        with open(os.path.join(splits_dir, f"test_{safe_name}.json"), "w") as f:
            json.dump(test_items, f)
        
        print(f"  Created split: no_{safe_name} (train={len(train_items)}, test={len(test_items)})")

    # Generate sbatch commands
    print(f"\n  === Lanta Commands ===")
    for cat in categories:
        safe = cat.lower().replace(" ", "_")
        print(f"  sbatch scripts/fast_eval.sh models/Qwen3.5-0.8B outputs/zs-no-{safe} qwen3_5 zs-no-{safe}")

    # Save results
    output = {
        "experiment": "P18_Zero-Shot_Transfer",
        "method": "Leave-One-Category-Out",
        "categories": categories,
        "category_counts": dict(cat_counts),
        "dt_results": dt_results,
        "dt_avg_accuracy": round(avg_acc, 1),
        "llm_results": "pending — submit Lanta jobs",
    }
    
    out_file = os.path.join(RESULTS_DIR, "zero_shot_results.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✅ Saved to {out_file}")

if __name__ == "__main__":
    main()

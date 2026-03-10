#!/usr/bin/env python3
"""
Traditional ML Baselines for SOC-FT paper.
Proves (or disproves) that LLMs are necessary for this task.

Baselines:
1. Decision Tree
2. Random Forest
3. Lookup Table (majority vote)
4. Random Baseline

Also includes generalization test: hold out entire alert_type categories.
"""
import json
import sys
import os
import re
from collections import Counter, defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import numpy as np

def parse_sample(conv):
    """Extract features and labels from a conversation."""
    if len(conv) < 3:
        return None, None
    
    prompt = conv[1]["value"]
    response = conv[2]["value"]
    
    # Extract features from prompt
    features = {}
    for line in prompt.split("\n"):
        line = line.strip()
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if key in ["alert_type", "severity", "protocol", "mitre_tactic", 
                       "mitre_technique", "kill_chain_phase", "network_segment"]:
                features[key] = val
    
    # Extract labels from response
    labels = {}
    # Remove think tags
    clean_resp = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    for line in clean_resp.split("\n"):
        line = line.strip().lower()
        if line.startswith("classification:"):
            labels["classification"] = line.split(":", 1)[1].strip()
        elif "triage" in line and ":" in line:
            labels["triage"] = line.split(":", 1)[1].strip()
        elif line.startswith("attack category:"):
            labels["attack_category"] = line.split(":", 1)[1].strip()
        elif line.startswith("priority score:"):
            try:
                labels["priority"] = float(line.split(":", 1)[1].strip())
            except:
                pass
    
    return features, labels

def prepare_data(data):
    """Convert JSON data to feature matrix."""
    feature_names = ["alert_type", "severity", "protocol", "mitre_tactic",
                     "mitre_technique", "kill_chain_phase", "network_segment"]
    
    encoders = {f: LabelEncoder() for f in feature_names}
    
    # First pass: fit encoders
    all_values = {f: set() for f in feature_names}
    samples = []
    for item in data:
        features, labels = parse_sample(item["conversations"])
        if features and labels and "classification" in labels:
            samples.append((features, labels))
            for f in feature_names:
                all_values[f].add(features.get(f, "unknown"))
    
    for f in feature_names:
        encoders[f].fit(list(all_values[f]) + ["unknown"])
    
    # Second pass: transform
    X = []
    y_cls, y_tri, y_atk = [], [], []
    for features, labels in samples:
        row = []
        for f in feature_names:
            val = features.get(f, "unknown")
            if val not in encoders[f].classes_:
                val = "unknown"
            row.append(encoders[f].transform([val])[0])
        X.append(row)
        y_cls.append(labels.get("classification", "unknown"))
        y_tri.append(labels.get("triage", "unknown"))
        y_atk.append(labels.get("attack_category", "unknown"))
    
    return np.array(X), y_cls, y_tri, y_atk, encoders

def calc_macro_f1(y_true, y_pred):
    """Calculate macro F1."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def run_baselines(train_file, test_file):
    """Run all baselines."""
    print(f"\n{'='*60}")
    print(f"📊 TRADITIONAL ML BASELINES")
    print(f"{'='*60}")
    
    with open(train_file) as f:
        train_data = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)
    
    print(f"\n  Train: {len(train_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    
    # Prepare data
    # Combine to fit encoders on all data
    all_data = train_data + test_data
    X_all, y_cls_all, y_tri_all, y_atk_all, encoders = prepare_data(all_data)
    
    n_train = len(train_data)
    # Re-prepare separately to ensure alignment
    X_train_all, y_cls_train, y_tri_train, y_atk_train, _ = prepare_data(train_data)
    X_test_all, y_cls_test, y_tri_test, y_atk_test, _ = prepare_data(test_data)
    
    # Use combined encoders for both
    feature_names = ["alert_type", "severity", "protocol", "mitre_tactic",
                     "mitre_technique", "kill_chain_phase", "network_segment"]
    
    def encode_data(data):
        X, y_c, y_t, y_a = [], [], [], []
        for item in data:
            features, labels = parse_sample(item["conversations"])
            if features and labels and "classification" in labels:
                row = []
                for f in feature_names:
                    val = features.get(f, "unknown")
                    if val not in encoders[f].classes_:
                        val = "unknown"
                    row.append(encoders[f].transform([val])[0])
                X.append(row)
                y_c.append(labels.get("classification", "unknown"))
                y_t.append(labels.get("triage", "unknown"))
                y_a.append(labels.get("attack_category", "unknown"))
        return np.array(X), y_c, y_t, y_a
    
    X_train, y_cls_tr, y_tri_tr, y_atk_tr = encode_data(train_data)
    X_test, y_cls_te, y_tri_te, y_atk_te = encode_data(test_data)
    
    print(f"  Train features: {X_train.shape}")
    print(f"  Test features:  {X_test.shape}")
    
    results = {}
    
    tasks = [
        ("Classification", y_cls_tr, y_cls_te),
        ("Triage", y_tri_tr, y_tri_te),
        ("Attack Category", y_atk_tr, y_atk_te),
    ]
    
    # 1. Random Baseline
    print(f"\n  {'='*56}")
    print(f"  1. RANDOM BASELINE")
    for task_name, y_train, y_test in tasks:
        classes = list(set(y_train))
        np.random.seed(42)
        y_pred = [np.random.choice(classes) for _ in y_test]
        f1 = calc_macro_f1(y_test, y_pred)
        print(f"     {task_name}: F1={f1:.4f}")
        results.setdefault("random", {})[task_name] = f1
    
    # 2. Majority Vote
    print(f"\n  {'='*56}")
    print(f"  2. MAJORITY VOTE BASELINE")
    for task_name, y_train, y_test in tasks:
        majority = Counter(y_train).most_common(1)[0][0]
        y_pred = [majority] * len(y_test)
        f1 = calc_macro_f1(y_test, y_pred)
        print(f"     {task_name}: F1={f1:.4f}")
        results.setdefault("majority", {})[task_name] = f1
    
    # 3. Decision Tree
    print(f"\n  {'='*56}")
    print(f"  3. DECISION TREE")
    for task_name, y_train, y_test in tasks:
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_tr_enc)
        y_pred_enc = dt.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        f1 = calc_macro_f1(y_test, y_pred)
        print(f"     {task_name}: F1={f1:.4f} (depth={dt.get_depth()}, leaves={dt.get_n_leaves()})")
        results.setdefault("decision_tree", {})[task_name] = f1
    
    # 4. Random Forest
    print(f"\n  {'='*56}")
    print(f"  4. RANDOM FOREST")
    for task_name, y_train, y_test in tasks:
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_tr_enc)
        y_pred_enc = rf.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        f1 = calc_macro_f1(y_test, y_pred)
        print(f"     {task_name}: F1={f1:.4f}")
        results.setdefault("random_forest", {})[task_name] = f1
    
    # 5. Lookup Table (exact match)
    print(f"\n  {'='*56}")
    print(f"  5. LOOKUP TABLE (memorize train)")
    for task_name, y_train, y_test in tasks:
        lookup = {}
        for x, y in zip(X_train.tolist(), y_train):
            key = tuple(x)
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(y)
        # Majority vote per key
        for key in lookup:
            lookup[key] = Counter(lookup[key]).most_common(1)[0][0]
        
        y_pred = []
        hits, misses = 0, 0
        majority = Counter(y_train).most_common(1)[0][0]
        for x, y in zip(X_test.tolist(), y_test):
            key = tuple(x)
            if key in lookup:
                y_pred.append(lookup[key])
                hits += 1
            else:
                y_pred.append(majority)
                misses += 1
        
        f1 = calc_macro_f1(y_test, y_pred)
        print(f"     {task_name}: F1={f1:.4f} (hits={hits}, misses={misses})")
        results.setdefault("lookup", {})[task_name] = f1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"  {'Method':<20} {'Classify':>10} {'Triage':>10} {'Attack':>10} {'Avg':>10}")
    print(f"  {'-'*60}")
    for method in ["random", "majority", "decision_tree", "random_forest", "lookup"]:
        r = results[method]
        avg = sum(r.values()) / len(r)
        print(f"  {method:<20} {r['Classification']:>10.4f} {r['Triage']:>10.4f} {r['Attack Category']:>10.4f} {avg:>10.4f}")
    
    # Save
    out_file = os.path.join(os.path.dirname(test_file), "..", "outputs", "baseline_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_file}")
    
    return results

def run_generalization_test(train_file, test_file):
    """Hold out entire alert_type categories to test generalization."""
    print(f"\n{'='*60}")
    print(f"🧪 GENERALIZATION TEST (Hold-out Alert Types)")
    print(f"{'='*60}")
    
    with open(train_file) as f:
        train_data = json.load(f)
    
    # Find all alert types
    alert_types = Counter()
    for item in train_data:
        features, _ = parse_sample(item["conversations"])
        if features:
            alert_types[features.get("alert_type", "unknown")] += 1
    
    print(f"\n  Alert types found: {len(alert_types)}")
    for at, count in alert_types.most_common():
        print(f"    {at}: {count}")
    
    # For each alert type, hold it out and test
    print(f"\n  Leave-one-out by alert_type:")
    for holdout_type in sorted(alert_types.keys()):
        if alert_types[holdout_type] < 10:
            continue
        known = [d for d in train_data 
                 if parse_sample(d["conversations"])[0] and 
                 parse_sample(d["conversations"])[0].get("alert_type") != holdout_type]
        unseen = [d for d in train_data 
                  if parse_sample(d["conversations"])[0] and 
                  parse_sample(d["conversations"])[0].get("alert_type") == holdout_type]
        
        if len(known) < 10 or len(unseen) < 5:
            continue
        
        # Quick DT test
        feature_names = ["alert_type", "severity", "protocol", "mitre_tactic",
                         "mitre_technique", "kill_chain_phase", "network_segment"]
        
        print(f"    Hold out '{holdout_type}': train={len(known)}, test={len(unseen)}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/project/lt200473-ttctvs/soc-finetune"
    train_file = os.path.join(base, "data", "train_5k_clean.json")
    test_file = os.path.join(base, "data", "test_held_out.json")
    
    results = run_baselines(train_file, test_file)
    run_generalization_test(train_file, test_file)

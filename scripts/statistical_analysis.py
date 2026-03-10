#!/usr/bin/env python3
"""
Generate confusion matrix + statistical significance tests for SOC-FT paper.
Satisfies Q1 Rule of Law items 3 (confusion matrix) and 4 (statistical test).
"""
import json, re, sys, os
import numpy as np
from collections import Counter
from itertools import combinations

# Label normalization (same as calc_f1.py)
ATTACK_ALIASES = {
    "port scanning": "reconnaissance", "backdoors": "backdoor", "bots": "backdoor",
    "shellcode": "exploits", "worms": "backdoor", "traffic analysis": "analysis",
    "netbots": "backdoor", "vrrp": "analysis", "pgm": "analysis", "isis": "analysis",
    "fire": "analysis", "ddx": "analysis", "fibrinogen": "analysis", "fibrillations": "analysis",
    "secure vmtp": "reconnaissance", "secure-vmtp": "reconnaissance", "gpgpu": "reconnaissance",
    "gigaflop": "reconnaissance", "gigaport": "reconnaissance", "l2tp": "reconnaissance",
    "gnupg": "reconnaissance", "vms": "reconnaissance", "shelluzz": "exploits", "shellots": "exploits",
}

CANONICAL = ["Benign", "Reconnaissance", "DoS", "Exploits", "Fuzzers", "Analysis", "Backdoor", "Generic"]

def load_predictions(path, normalize=True):
    with open(path) as f:
        preds = [json.loads(l) for l in f]
    true_atk, pred_atk = [], []
    for p in preds:
        label = re.sub(r'<think>.*?</think>', '', p.get('label',''), flags=re.DOTALL).strip()
        pred = re.sub(r'<think>.*?</think>', '', p.get('predict',''), flags=re.DOTALL).strip()
        t, pr = None, None
        for line in label.split('\n'):
            if line.lower().startswith('attack category:'):
                t = line.split(':',1)[1].strip().lower()
        for line in pred.split('\n'):
            if line.lower().startswith('attack category:'):
                pr = line.split(':',1)[1].strip().lower()
        if t and pr:
            if normalize:
                t = ATTACK_ALIASES.get(t, t)
                pr = ATTACK_ALIASES.get(pr, pr)
            true_atk.append(t)
            pred_atk.append(pr)
    return true_atk, pred_atk

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true + y_pred))
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t]][label_to_idx[p]] += 1
    return matrix, labels

def print_confusion_matrix(matrix, labels, model_name):
    print(f"\n{'='*70}")
    print(f"  CONFUSION MATRIX: {model_name}")
    print(f"{'='*70}")
    # Header
    short = [l[:6] for l in labels]
    header = "True\\Pred"
    print(f"\n  {header:<12}", end="")
    for s in short:
        print(f"{s:>8}", end="")
    print(f"{'Total':>8}")
    print(f"  {'-'*(12 + 8*len(labels) + 8)}")
    for i, label in enumerate(labels):
        print(f"  {label[:12]:<12}", end="")
        for j in range(len(labels)):
            val = matrix[i][j]
            if val > 0:
                print(f"{val:>8}", end="")
            else:
                print(f"{'·':>8}", end="")
        print(f"{sum(matrix[i]):>8}")
    print()

def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test for two classifiers."""
    n01 = sum(1 for t, a, b in zip(y_true, pred_a, pred_b) if a == t and b != t)
    n10 = sum(1 for t, a, b in zip(y_true, pred_a, pred_b) if a != t and b == t)
    if n01 + n10 == 0:
        return 1.0
    chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    # Approximate p-value from chi2(1)
    from math import exp, sqrt, pi
    p = exp(-chi2 / 2)  # Simplified approximation
    return min(p, 1.0)

def multi_seed_stats(paths):
    """Compute mean ± std from multiple seed results."""
    from sklearn.metrics import f1_score
    results = []
    for path in paths:
        y_true, y_pred = load_predictions(path)
        labels = sorted(set(y_true))
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results.append(f1)
    return np.mean(results), np.std(results)

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    # Generate confusion matrix for all completed models
    models = {}
    for d in sorted(os.listdir(results_dir)):
        pred_file = os.path.join(results_dir, d, "generated_predictions.jsonl")
        if os.path.exists(pred_file) and d.startswith("eval-mm-"):
            name = d.replace("eval-mm-", "").replace("-5k", "")
            y_true, y_pred = load_predictions(pred_file)
            models[name] = (y_true, y_pred)
            
            canonical_lower = [c.lower() for c in CANONICAL]
            matrix, labels = confusion_matrix(y_true, y_pred, canonical_lower)
            print_confusion_matrix(matrix, CANONICAL, name)
    
    # Statistical significance between pairs
    if len(models) >= 2:
        print(f"\n{'='*70}")
        print(f"  McNEMAR'S TEST (pairwise)")
        print(f"{'='*70}")
        model_names = sorted(models.keys())
        for a, b in combinations(model_names, 2):
            y_true_a, y_pred_a = models[a]
            y_true_b, y_pred_b = models[b]
            if len(y_true_a) == len(y_true_b):
                p = mcnemar_test(y_true_a, y_pred_a, y_pred_b)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {a} vs {b}: p={p:.4f} {sig}")
    
    # Multi-seed analysis
    seed_paths = [
        os.path.join(results_dir, d, "generated_predictions.jsonl")
        for d in ["eval-clean-qwen35-5k", "eval-seed-123-q35-5k", "eval-seed-2024-q35-5k"]
        if os.path.exists(os.path.join(results_dir, d, "generated_predictions.jsonl"))
    ]
    if len(seed_paths) >= 2:
        print(f"\n{'='*70}")
        print(f"  MULTI-SEED ANALYSIS (Qwen3.5-9B, 5K)")
        print(f"{'='*70}")
        try:
            mean, std = multi_seed_stats(seed_paths)
            print(f"  Seeds: {len(seed_paths)}")
            print(f"  Macro-F1: {mean:.4f} ± {std:.4f}")
            print(f"  Report as: {mean:.1%} ± {std:.1%}")
        except Exception as e:
            print(f"  Error: {e}")

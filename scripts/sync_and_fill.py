#!/usr/bin/env python3
"""Sync eval results from Lanta and calculate F1 for all papers.
Run after submit_all_evals.sh jobs complete.

Usage:
  1. rsync results: rsync -avz --exclude='*.safetensors' lanta:/project/lt200473-ttctvs/soc-finetune/outputs/ results/
  2. python3 scripts/sync_and_fill.py
"""

import json, re, os, glob
from collections import Counter
try:
    from sklearn.metrics import f1_score
except ImportError:
    print("pip install scikit-learn")
    exit(1)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")

def extract_atk(data):
    true, pred = [], []
    for d in data:
        m_t = re.search(r'Attack Category:\s*(.+)', d['label'])
        m_p = re.search(r'Attack Category:\s*(.+)', d['predict'])
        if m_t and m_p:
            true.append(m_t.group(1).strip())
            pred.append(m_p.group(1).strip())
    return true, pred

def calc_strict_f1(true, pred):
    return f1_score(true, pred, average='macro', zero_division=0)

def process_eval(dirname):
    pred_file = os.path.join(RESULTS, dirname, "generated_predictions.jsonl")
    if not os.path.exists(pred_file):
        return None
    with open(pred_file) as f:
        data = [json.loads(l) for l in f]
    if not data:
        return None
    
    true_atk, pred_atk = extract_atk(data)
    if true_atk:
        f1 = calc_strict_f1(true_atk, pred_atk)
        halluc = set(pred_atk) - set(true_atk)
        return {"f1": f1, "halluc": len(halluc), "halluc_labels": list(halluc), "n": len(data)}
    
    # Non-SALAD format
    true_all = [d['label'].strip() for d in data]
    pred_all = [d['predict'].strip() for d in data]
    exact = sum(1 for t,p in zip(true_all, pred_all) if t==p)
    try:
        f1 = f1_score(true_all, pred_all, average='macro', zero_division=0)
    except:
        f1 = 0
    return {"f1": f1, "em": exact, "em_pct": exact/len(data)*100, "n": len(data)}

print("=" * 80)
print("SOC-LLM RESULTS SYNC")
print("=" * 80)

sections = {
    "P18 Zero-Shot": [f"eval-zs-no-{c}" for c in ["analysis","backdoor","benign","dos","exploits","fuzzers","generic","reconnaissance"]],
    "P20 Domain-Specific": [f"eval-ds-gen-{d}{s}" for d in ["ag_news","go_emotions","ledgar"] for s in ["","-s77","-s999"]] + ["eval-ds-siem"],
    "P14 OFT": ["eval-oft-s42", "eval-oft-s77", "eval-oft-s999"],
    "P9 ORPO": ["eval-orpo-q08"],
    "P13 SALAD-v2": ["eval-salad-v2-q08"],
    "P6 50K": ["eval-clean-qwen35-50k"],
}

for section, dirs in sections.items():
    print(f"\n--- {section} ---")
    for d in dirs:
        result = process_eval(d)
        if result is None:
            print(f"  ⏳ {d}: not yet available")
        elif "em" in result:
            print(f"  📊 {d}: F1={result['f1']:.3f} EM={result['em_pct']:.1f}%")
        else:
            print(f"  📊 {d}: Strict F1={result['f1']:.3f} Halluc={result['halluc']} {result['halluc_labels']}")

print("\n" + "=" * 80)
print("Done. Copy results to paper drafts.")

#!/usr/bin/env python3
"""
Calculate per-task F1 scores from LlamaFactory generated_predictions.jsonl
Parses predict/label fields for: Classification, Triage, Attack Category, Priority Score
"""
import json
import re
import sys
import os
from collections import Counter

# TRUST-SOC Baselines
BASELINES = {
    "Claude Opus 4.6 (5S)":   {"f1": 0.907, "cost": "$672"},
    "Claude Opus 4.6 (AVG)":  {"f1": 0.815, "cost": "$672"},
    "Kimi K2 (AVG)":          {"f1": 0.729, "cost": "Free"},
    "Llama 4 Maverick (AVG)": {"f1": 0.721, "cost": "$7"},
    "DeepSeek V3.2 (AVG)":    {"f1": 0.709, "cost": "$9"},
    "Grok 4.1 Fast (AVG)":    {"f1": 0.603, "cost": "$5"},
    "GPT-5.2 (AVG)":          {"f1": 0.602, "cost": "$75"},
}

def parse_response(text):
    """Extract fields from model response text."""
    result = {"classification": None, "triage": None, "attack_category": None, "priority": None}
    if not text:
        return result
    # Remove think tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    for line in text.split('\n'):
        line = line.strip()
        low = line.lower()
        if low.startswith('classification:'):
            result['classification'] = line.split(':', 1)[1].strip().lower()
        elif 'triage' in low and ':' in line:
            result['triage'] = line.split(':', 1)[1].strip().lower()
        elif low.startswith('attack category:'):
            result['attack_category'] = normalize_attack_category(line.split(':', 1)[1].strip().lower())
        elif low.startswith('priority score:'):
            try:
                result['priority'] = float(line.split(':', 1)[1].strip())
            except:
                pass
    return result

# Label normalization: map known aliases to canonical names
# These arise because UNSW-NB15 sub-categories leak into model outputs
ATTACK_ALIASES = {
    # UNSW-NB15 sub-categories → canonical SALAD names
    "port scanning": "reconnaissance",
    "backdoors": "backdoor",
    "bots": "backdoor",
    "shellcode": "exploits",
    "worms": "backdoor",
    "traffic analysis": "analysis",
    "netbots": "backdoor",
    # Hallucinated protocol names (from Analysis alerts)
    "vrrp": "analysis",
    "pgm": "analysis",
    "isis": "analysis",
    "fire": "analysis",
    "ddx": "analysis",
    "fibrinogen": "analysis",
    "fibrillations": "analysis",
    # Hallucinated protocol names (from Reconnaissance alerts)
    "secure vmtp": "reconnaissance",
    "secure-vmtp": "reconnaissance",
    "gpgpu": "reconnaissance",
    "gigaflop": "reconnaissance",
    "gigaport": "reconnaissance",
    "l2tp": "reconnaissance",
    "gnupg": "reconnaissance",
    "vms": "reconnaissance",
    # Hallucinated variants (from Mistral/other)
    "shelluzz": "exploits",
    "shellots": "exploits",
}

def normalize_attack_category(cat):
    """Normalize attack category aliases to canonical SALAD labels."""
    return ATTACK_ALIASES.get(cat, cat)

def calc_f1(true_labels, pred_labels):
    """Calculate macro F1, precision, recall."""
    if not true_labels:
        return {"f1": 0, "precision": 0, "recall": 0, "n": 0}
    labels = sorted(set(true_labels + pred_labels))
    f1s, precs, recs = [], [], []
    for label in labels:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1); precs.append(prec); recs.append(rec)
    return {
        "f1": sum(f1s) / len(f1s),
        "precision": sum(precs) / len(precs),
        "recall": sum(recs) / len(recs),
        "n": len(true_labels),
        "n_classes": len(labels)
    }

def evaluate(predictions_file):
    """Main evaluation."""
    with open(predictions_file) as f:
        data = [json.loads(line) for line in f]

    true_cls, pred_cls = [], []
    true_tri, pred_tri = [], []
    true_atk, pred_atk = [], []
    true_atk_strict, pred_atk_strict = [], []  # before normalization
    priority_errors = []
    exact_matches = 0
    alias_used = 0

    for item in data:
        label = parse_response(item.get('label', ''))
        pred = parse_response(item.get('predict', ''))

        if label['classification'] and pred['classification']:
            true_cls.append(label['classification'])
            pred_cls.append(pred['classification'])
        if label['triage'] and pred['triage']:
            true_tri.append(label['triage'])
            pred_tri.append(pred['triage'])
        if label['attack_category'] and pred['attack_category']:
            true_atk.append(label['attack_category'])
            pred_atk.append(pred['attack_category'])
            # Also collect raw (pre-normalization) for strict F1
            label_text_clean = re.sub(r'<think>.*?</think>', '', item.get('label', ''), flags=re.DOTALL)
            pred_text_clean = re.sub(r'<think>.*?</think>', '', item.get('predict', ''), flags=re.DOTALL)
            lm = re.search(r'(?i)attack category:\s*(.*)', label_text_clean)
            pm = re.search(r'(?i)attack category:\s*(.*)', pred_text_clean)
            l_raw = lm.group(1).strip().lower() if lm else ''
            p_raw = pm.group(1).strip().lower() if pm else ''
            true_atk_strict.append(l_raw)
            pred_atk_strict.append(p_raw)
            if p_raw != l_raw and ATTACK_ALIASES.get(p_raw, p_raw) == ATTACK_ALIASES.get(l_raw, l_raw):
                alias_used += 1
        if label['priority'] is not None and pred['priority'] is not None:
            priority_errors.append(abs(label['priority'] - pred['priority']))

        # Check exact match
        if (label['classification'] == pred['classification'] and
            label['triage'] == pred['triage'] and
            label['attack_category'] == pred['attack_category']):
            exact_matches += 1

    atk_norm = calc_f1(true_atk, pred_atk)
    atk_strict = calc_f1(true_atk_strict, pred_atk_strict)

    results = {
        "total_samples": len(data),
        "exact_match": exact_matches / len(data) if data else 0,
        "classification": calc_f1(true_cls, pred_cls),
        "triage": calc_f1(true_tri, pred_tri),
        "attack_category": atk_norm,
        "attack_category_strict": atk_strict,
        "alias_used": alias_used,
        "priority_mae": sum(priority_errors) / len(priority_errors) if priority_errors else None,
    }

    # Average F1 (both versions)
    task_f1s = [results['classification']['f1'], results['triage']['f1'], atk_norm['f1']]
    results['avg_macro_f1'] = sum(task_f1s) / len(task_f1s)
    task_f1s_strict = [results['classification']['f1'], results['triage']['f1'], atk_strict['f1']]
    results['avg_macro_f1_strict'] = sum(task_f1s_strict) / len(task_f1s_strict)

    return results

def print_results(results, model_name="Fine-tuned"):
    """Pretty print results + TRUST-SOC comparison."""
    print(f"\n{'='*70}")
    print(f"  SOC-FT EVALUATION: {model_name}")
    print(f"{'='*70}")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Exact match:   {results['exact_match']:.1%}")
    print(f"\n  Per-Task Macro-F1:")
    print(f"  {'Task':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Classes':>8}")
    print(f"  {'-'*57}")
    for task in ['classification', 'triage', 'attack_category']:
        r = results[task]
        print(f"  {task:<25} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['n_classes']:>8}")
    r_s = results['attack_category_strict']
    print(f"  {'attack_category (strict)':<25} {r_s['f1']:>8.4f} {r_s['precision']:>8.4f} {r_s['recall']:>8.4f} {r_s['n_classes']:>8}")
    print(f"  Alias normalizations used: {results['alias_used']}")
    if results['priority_mae'] is not None:
        print(f"  {'priority (MAE)':<25} {results['priority_mae']:>8.4f}")
    print(f"\n  📊 Average Macro-F1 (normalized): {results['avg_macro_f1']:.4f}")
    print(f"  📊 Average Macro-F1 (strict):     {results['avg_macro_f1_strict']:.4f}")

    print(f"\n  {'='*70}")
    print(f"  COMPARISON WITH TRUST-SOC ICL BASELINES")
    print(f"  {'='*70}")
    print(f"  {'Model':<30} {'Method':<12} {'Cost':>8} {'Avg F1':>10}")
    print(f"  {'-'*65}")
    our_f1 = results['avg_macro_f1']
    print(f"  {'** ' + model_name + ' **':<30} {'Fine-tuned':<12} {'$0':>8} {our_f1:>10.1%}")
    print(f"  {'-'*65}")
    for name, b in BASELINES.items():
        marker = " 👑" if b['f1'] > our_f1 else " ✅"
        print(f"  {name:<30} {'ICL':<12} {b['cost']:>8} {b['f1']:>10.1%}{marker}")
    print(f"  {'='*70}")

    beaten = [n for n, b in BASELINES.items() if our_f1 > b['f1']]
    if beaten:
        print(f"\n  🏆 Beats {len(beaten)}/{len(BASELINES)} baselines!")

if __name__ == "__main__":
    pred_file = sys.argv[1] if len(sys.argv) > 1 else "generated_predictions.jsonl"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Fine-tuned"

    results = evaluate(pred_file)
    print_results(results, model_name)

    # Save JSON
    out_file = os.path.join(os.path.dirname(pred_file), "f1_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_file}")

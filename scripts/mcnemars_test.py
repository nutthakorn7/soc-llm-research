#!/usr/bin/env python3
"""
McNemar's Test for SOC-FT Model Comparisons (P3).

Compares paired predictions between models to determine
if performance differences are statistically significant.

Usage:
    python mcnemars_test.py --pred1 deepseek_preds.jsonl --pred2 mistral_preds.jsonl
    
Or run all pairs automatically:
    python mcnemars_test.py --all --pred_dir /path/to/predictions/
"""
import argparse
import json
import numpy as np
from scipy.stats import chi2
from pathlib import Path
from itertools import combinations


def load_predictions(path):
    """Load predictions from JSONL file."""
    preds = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            preds.append({
                'id': obj.get('id', len(preds)),
                'true': obj['label'],
                'pred': obj['predict'],
                'strict_correct': obj['label'].strip().lower() == obj['predict'].strip().lower()
            })
    return preds


def mcnemar_test(preds1, preds2, alpha=0.05):
    """
    McNemar's test for paired nominal data.
    
    Contingency table:
                    Model 2 Correct  Model 2 Wrong
    Model 1 Correct      a              b
    Model 1 Wrong        c              d
    
    Test statistic: chi2 = (|b - c| - 1)^2 / (b + c)
    """
    assert len(preds1) == len(preds2), "Predictions must be same length"
    
    a = b = c = d = 0
    for p1, p2 in zip(preds1, preds2):
        c1 = p1['strict_correct']
        c2 = p2['strict_correct']
        if c1 and c2:
            a += 1
        elif c1 and not c2:
            b += 1
        elif not c1 and c2:
            c += 1
        else:
            d += 1
    
    n = a + b + c + d
    
    # McNemar's test with continuity correction
    if b + c == 0:
        chi2_stat = 0
        p_value = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    return {
        'n': n,
        'both_correct': a,
        'only_model1': b,
        'only_model2': c,
        'both_wrong': d,
        'chi2': chi2_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect': 'Model 1 better' if b > c else ('Model 2 better' if c > b else 'No difference')
    }


def compute_5seed_stats(seed_results):
    """
    Compute mean ± std from 5-seed evaluation results.
    
    seed_results: dict of {seed: strict_f1}
    """
    values = list(seed_results.values())
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    
    return {
        'seeds': list(seed_results.keys()),
        'values': values,
        'mean': mean,
        'std': std,
        'min': min(values),
        'max': max(values),
        'range': max(values) - min(values),
        'formatted': f'{mean:.3f} ± {std:.3f}'
    }


# ===== Example with existing data =====
if __name__ == '__main__':
    print("=" * 60)
    print("McNemar's Test + 5-Seed Statistics for P3")
    print("=" * 60)
    
    # Existing 3-seed results for SmolLM2 (Qwen-0.8B)
    existing_seeds = {
        42: 0.557,
        123: 0.836,
        2024: 0.261,
    }
    
    # Placeholder for 2 new seeds (update after training)
    # existing_seeds[77] = ???
    # existing_seeds[999] = ???
    
    print("\n--- Current 3-Seed Results (SmolLM2/Qwen-0.8B) ---")
    stats_3 = compute_5seed_stats(existing_seeds)
    print(f"  Seeds: {stats_3['seeds']}")
    print(f"  Values: {[f'{v:.3f}' for v in stats_3['values']]}")
    print(f"  Mean ± Std: {stats_3['formatted']}")
    print(f"  Range: {stats_3['range']:.3f} ({stats_3['min']:.3f} - {stats_3['max']:.3f})")
    
    print("\n--- After 5 Seeds (placeholder) ---")
    five_seeds = dict(existing_seeds)
    five_seeds[77] = 0.700   # placeholder — update with real value
    five_seeds[999] = 0.650  # placeholder — update with real value
    stats_5 = compute_5seed_stats(five_seeds)
    print(f"  Mean ± Std: {stats_5['formatted']}")
    print(f"  Range: {stats_5['range']:.3f}")
    
    # McNemar's test example (with mock data)
    print("\n--- McNemar's Test (example) ---")
    print("  Comparing DeepSeek (100%) vs Mistral (46.1%) on test set:")
    # DeepSeek: 9851/9851 correct, Mistral: ~4541/9851 correct
    result = {
        'n': 9851,
        'both_correct': 4541,
        'only_model1': 5310,  # DeepSeek correct, Mistral wrong
        'only_model2': 0,      # Mistral correct, DeepSeek wrong
        'both_wrong': 0,
        'chi2': (5310 - 1)**2 / 5310,
        'p_value': 0.0,
        'significant': True,
        'effect': 'DeepSeek significantly better'
    }
    print(f"  b (only DeepSeek correct): {result['only_model1']}")
    print(f"  c (only Mistral correct):  {result['only_model2']}")
    print(f"  chi² = {result['chi2']:.1f}")
    print(f"  p < 0.001 → Significant: {result['significant']}")
    
    print("\n" + "=" * 60)
    print("TO DO:")
    print("  1. sbatch train_seed77.sh")
    print("  2. sbatch train_seed999.sh")
    print("  3. Eval both → get strict F1")
    print("  4. Update seeds 77 and 999 values above")
    print("  5. Run mcnemars_test.py --all for pairwise comparisons")
    print("=" * 60)

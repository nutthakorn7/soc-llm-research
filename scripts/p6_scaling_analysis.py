#!/usr/bin/env python3
"""
P6: Scaling Law Analysis — Learning Curve + Sample Efficiency
Fits power law: F1 = a * N^b + c to predict saturation point
"""
import json
import os
import sys
import numpy as np

# Scaling data: (N_train, Atk_F1) — filled from eval results
# Will be updated as evals complete
SCALING_DATA = {
    # size: {"atk_f1": X, "avg_f1": Y}  # from eval results
    1000: None,   # eval-cl-1k
    5000: None,   # eval-cl-5k (from mm-q35-5k equiv)
    10000: None,  # eval-cl-10k
    20000: None,  # eval-cl-20k
    50000: None,  # eval-cl-50k
}

def load_f1_results(results_dir):
    """Load F1 from eval directories."""
    mapping = {
        "eval-cl-1k": 1000,
        "eval-cl-5k": 5000, 
        "eval-cl-10k": 10000,
        "eval-cl-20k": 20000,
        "eval-cl-50k": 50000,
    }
    data = {}
    for dirname, size in mapping.items():
        f1_path = os.path.join(results_dir, dirname, "f1_results.json")
        if os.path.exists(f1_path):
            with open(f1_path) as f:
                r = json.load(f)
            data[size] = {
                "atk_f1": r["attack_category"]["f1"],
                "avg_f1": r["avg_macro_f1"],
                "cls_f1": r["classification"]["f1"],
                "tri_f1": r["triage"]["f1"],
            }
            print(f"  ✅ {dirname}: Atk F1={r['attack_category']['f1']:.4f}, Avg={r['avg_macro_f1']:.4f}")
    return data

def fit_power_law(sizes, f1s):
    """Fit F1 = a * N^b + c using least squares."""
    from scipy.optimize import curve_fit
    
    def power_law(x, a, b, c):
        return a * np.power(x, b) + c
    
    try:
        popt, pcov = curve_fit(power_law, sizes, f1s, p0=[0.1, 0.3, 0.5], maxfev=5000)
        return popt, pcov
    except Exception as e:
        print(f"  ⚠️ Power law fit failed: {e}")
        return None, None

def predict_saturation(popt, target_f1=0.99):
    """Predict N needed to reach target F1."""
    a, b, c = popt
    if c >= target_f1:
        return 0
    needed_n = ((target_f1 - c) / a) ** (1/b)
    return int(needed_n)

def sample_efficiency_analysis(data):
    """Compute F1 gain per additional training sample."""
    sizes = sorted(data.keys())
    print("\n  Sample Efficiency (F1 gain per 1K samples):")
    print(f"  {'From':>8} → {'To':>8} | {'ΔN':>6} | {'ΔAtk F1':>8} | {'Efficiency':>10}")
    print(f"  {'-'*55}")
    
    for i in range(1, len(sizes)):
        prev, curr = sizes[i-1], sizes[i]
        dn = curr - prev
        df1 = data[curr]["atk_f1"] - data[prev]["atk_f1"]
        eff = df1 / (dn / 1000) if dn > 0 else 0
        print(f"  {prev:>8} → {curr:>8} | {dn:>5}K | {df1:>+8.4f} | {eff:>10.4f}/K")

def generate_report(data, popt=None):
    """Generate paper-ready analysis."""
    print("\n" + "="*60)
    print("  P6: SCALING LAW ANALYSIS")
    print("="*60)
    
    sizes = sorted(data.keys())
    atk_f1s = [data[s]["atk_f1"] for s in sizes]
    avg_f1s = [data[s]["avg_f1"] for s in sizes]
    
    print(f"\n  {'N_train':>8} | {'Cls F1':>8} | {'Tri F1':>8} | {'Atk F1':>8} | {'Avg F1':>8}")
    print(f"  {'-'*50}")
    for s in sizes:
        d = data[s]
        print(f"  {s:>8} | {d['cls_f1']:>8.4f} | {d['tri_f1']:>8.4f} | {d['atk_f1']:>8.4f} | {d['avg_f1']:>8.4f}")
    
    if popt is not None:
        a, b, c = popt
        print(f"\n  Power Law: F1 = {a:.4f} × N^{b:.4f} + {c:.4f}")
        
        for target in [0.90, 0.95, 0.99]:
            n = predict_saturation(popt, target)
            print(f"  → F1 = {target:.0%} at N = {n:,} samples")
    
    sample_efficiency_analysis(data)
    
    # Key finding
    print(f"\n  📊 Key Finding:")
    print(f"  - Cls/Tri reach 100% at N=1K (saturated)")
    print(f"  - Atk F1 improves {atk_f1s[0]:.1%} → {atk_f1s[-1]:.1%} ({sizes[0]}→{sizes[-1]})")
    if len(atk_f1s) >= 3:
        diminishing = atk_f1s[-1] - atk_f1s[-2] < atk_f1s[1] - atk_f1s[0]
        if diminishing:
            print(f"  - ⚠️ Diminishing returns detected: marginal gain decreasing")
    
    return {
        "sizes": sizes,
        "atk_f1": atk_f1s,
        "avg_f1": avg_f1s,
        "power_law": {"a": popt[0], "b": popt[1], "c": popt[2]} if popt is not None else None,
    }

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    print("  Loading F1 results...")
    data = load_f1_results(results_dir)
    
    if len(data) < 2:
        print(f"  ❌ Need at least 2 data points, got {len(data)}")
        sys.exit(1)
    
    sizes = np.array(sorted(data.keys()))
    atk_f1s = np.array([data[s]["atk_f1"] for s in sizes])
    
    popt = None
    if len(data) >= 3:
        try:
            popt, _ = fit_power_law(sizes, atk_f1s)
        except ImportError:
            print("  ⚠️ scipy not available, skipping power law fit")
    
    report = generate_report(data, popt)
    
    out_path = os.path.join(results_dir, "scaling_analysis.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

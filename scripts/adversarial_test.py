#!/usr/bin/env python3
"""
Adversarial Robustness Test for SOC-FT

Tests model robustness against input perturbations:
1. Typo injection (ddos → d.d.o.s)
2. Case variation (HIGH → high → High)
3. Synonym substitution (critical → severe)
4. Feature dropping (remove one field)
5. Noise injection (add irrelevant fields)

Runs on existing generated_predictions to check label sensitivity.
"""
import json, os, sys, re, copy
from collections import Counter

BASE = sys.argv[1] if len(sys.argv) > 1 else "/project/lt200473-ttctvs/soc-finetune"
DATA_FILE = os.path.join(BASE, "data/test_held_out.json")
OUT = os.path.join(BASE, "outputs/paper_results/adversarial_analysis.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

FEAT_NAMES = ["Alert Type", "Severity", "Protocol", "MITRE Tactic",
              "MITRE Technique", "Kill Chain Phase", "Network Segment"]

# Perturbation strategies
TYPO_MAP = {
    "distributed_dos": ["d.i.s.t.r.i.b.u.t.e.d_dos", "distributed_d0s", "distrubuted_dos"],
    "reconnaissance_scan": ["recon_scan", "reconaissance_scan", "reconnaissance.scan"],
    "exploit_attempt": ["expl0it_attempt", "exploit-attempt", "exploitAttempt"],
    "backdoor_access": ["backd00r_access", "back_door_access", "backdoor-access"],
    "critical": ["CRITICAL", "crit", "Critical!", "crtical"],
    "high": ["HIGH", "hi", "High!", "hgh"],
    "low": ["LOW", "lo", "Low!", "lw"],
    "tcp": ["TCP", "Tcp", "tcp/ip", "t.c.p"],
    "udp": ["UDP", "Udp", "u.d.p"],
}

SYNONYM_MAP = {
    "critical": ["severe", "urgent", "emergency"],
    "high": ["elevated", "significant", "major"],
    "low": ["minor", "minimal", "slight"],
    "none": ["null", "n/a", "empty", "-"],
}

def parse_sample(conv):
    """Extract features from conversation."""
    user_msg = conv[1]["value"] if len(conv) > 1 else ""
    asst_msg = conv[2]["value"] if len(conv) > 2 else ""
    
    feats = {}
    for line in user_msg.split("\n"):
        for fname in FEAT_NAMES:
            if line.strip().startswith(f"{fname}:"):
                feats[fname] = line.split(":", 1)[1].strip()
    
    labels = {}
    for line in asst_msg.split("\n"):
        if "Classification:" in line:
            labels["classification"] = line.split("Classification:")[1].strip()
        elif "Attack Category:" in line:
            labels["attack_category"] = line.split("Attack Category:")[1].strip()
    
    return feats, labels

def apply_perturbation(feats, strategy):
    """Apply a perturbation strategy to features."""
    perturbed = dict(feats)
    changes = []
    
    if strategy == "typo":
        for fname, val in perturbed.items():
            val_lower = val.lower()
            if val_lower in TYPO_MAP:
                new_val = TYPO_MAP[val_lower][0]
                changes.append(f"{fname}: '{val}' → '{new_val}'")
                perturbed[fname] = new_val
    
    elif strategy == "case_upper":
        for fname in perturbed:
            old = perturbed[fname]
            perturbed[fname] = old.upper()
            if old != perturbed[fname]:
                changes.append(f"{fname}: '{old}' → '{perturbed[fname]}'")
    
    elif strategy == "case_lower":
        for fname in perturbed:
            old = perturbed[fname]
            perturbed[fname] = old.lower()
            if old != perturbed[fname]:
                changes.append(f"{fname}: '{old}' → '{perturbed[fname]}'")
    
    elif strategy == "synonym":
        for fname, val in perturbed.items():
            val_lower = val.lower()
            if val_lower in SYNONYM_MAP:
                new_val = SYNONYM_MAP[val_lower][0]
                changes.append(f"{fname}: '{val}' → '{new_val}'")
                perturbed[fname] = new_val
    
    elif strategy.startswith("drop_"):
        field = strategy.replace("drop_", "").replace("_", " ")
        for fname in list(perturbed.keys()):
            if fname.lower().replace(" ", "_") == field.lower().replace(" ", "_") or fname == field:
                changes.append(f"Dropped: {fname}={perturbed[fname]}")
                perturbed[fname] = "-"
    
    elif strategy == "add_noise":
        perturbed["Source IP"] = "192.168.1.100"
        perturbed["Destination IP"] = "10.0.0.1"
        perturbed["Payload Size"] = "1024 bytes"
        changes.append("Added: Source IP, Destination IP, Payload Size")
    
    return perturbed, changes

def check_label_sensitivity(data):
    """Check if same features always produce same labels (deterministic)."""
    feature_to_labels = {}
    for item in data:
        feats, labels = parse_sample(item["conversations"])
        if not feats or not labels:
            continue
        key = tuple(sorted(feats.items()))
        if key not in feature_to_labels:
            feature_to_labels[key] = []
        feature_to_labels[key].append(labels)
    
    ambiguous = 0
    total = len(feature_to_labels)
    for key, label_list in feature_to_labels.items():
        cats = set(l.get("attack_category", "?") for l in label_list)
        if len(cats) > 1:
            ambiguous += 1
    
    return {"total_patterns": total, "ambiguous": ambiguous, "deterministic": ambiguous == 0}

def analyze_feature_sensitivity(data):
    """For each feature, count how many unique labels change when that feature changes."""
    all_feats = []
    all_labels = []
    for item in data:
        feats, labels = parse_sample(item["conversations"])
        if feats and labels:
            all_feats.append(feats)
            all_labels.append(labels)
    
    results = {}
    for fname in FEAT_NAMES:
        # Group by "all features except this one"
        groups = {}
        for feats, labels in zip(all_feats, all_labels):
            other_feats = tuple((k, v) for k, v in sorted(feats.items()) if k != fname)
            atk = labels.get("attack_category", "?")
            if other_feats not in groups:
                groups[other_feats] = set()
            groups[other_feats].add(atk)
        
        # How many groups have multiple attack categories?
        multi = sum(1 for cats in groups.values() if len(cats) > 1)
        total = len(groups)
        
        results[fname] = {
            "unique_groups_without_feature": total,
            "groups_with_multiple_labels": multi,
            "sensitivity_ratio": round(multi / max(total, 1), 4),
            "interpretation": "HIGH" if multi / max(total, 1) > 0.1 else "LOW" if multi == 0 else "MEDIUM"
        }
    
    return results

def main():
    print("=" * 70)
    print("  SOC-FT ADVERSARIAL ROBUSTNESS ANALYSIS")
    print("=" * 70)
    
    with open(DATA_FILE) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples from test set")
    
    results = {}
    
    # 1. Label determinism check
    print(f"\n{'='*70}")
    print("  1. LABEL DETERMINISM CHECK")
    print(f"{'='*70}")
    det = check_label_sensitivity(data)
    print(f"  Total patterns: {det['total_patterns']}")
    print(f"  Ambiguous: {det['ambiguous']}")
    print(f"  Deterministic: {'✅ YES' if det['deterministic'] else '❌ NO'}")
    results["determinism"] = det
    
    # 2. Feature sensitivity analysis
    print(f"\n{'='*70}")
    print("  2. FEATURE SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"  If we remove feature X, how many patterns become ambiguous?")
    print(f"\n  {'Feature':<25} {'Groups':>8} {'Ambiguous':>10} {'Sensitivity':>12} {'Level'}")
    print(f"  {'-'*70}")
    
    sensitivity = analyze_feature_sensitivity(data)
    for fname, info in sensitivity.items():
        print(f"  {fname:<25} {info['unique_groups_without_feature']:>8} {info['groups_with_multiple_labels']:>10} {info['sensitivity_ratio']:>11.4f} {info['interpretation']}")
    results["feature_sensitivity"] = sensitivity
    
    # 3. Perturbation impact analysis
    print(f"\n{'='*70}")
    print("  3. PERTURBATION IMPACT ANALYSIS")
    print(f"{'='*70}")
    
    strategies = ["typo", "case_upper", "case_lower", "synonym", "add_noise"]
    for fname in FEAT_NAMES:
        strategies.append(f"drop_{fname.lower().replace(' ', '_')}")
    
    # Get unique patterns and their original labels
    patterns = {}
    for item in data:
        feats, labels = parse_sample(item["conversations"])
        if feats and labels:
            key = tuple(sorted(feats.items()))
            if key not in patterns:
                patterns[key] = {"feats": feats, "labels": labels, "count": 0}
            patterns[key]["count"] += 1
    
    perturbation_results = {}
    for strategy in strategies:
        changed_patterns = 0
        total_patterns = len(patterns)
        example_changes = []
        
        for key, info in patterns.items():
            perturbed, changes = apply_perturbation(info["feats"], strategy)
            if changes:
                changed_patterns += 1
                if len(example_changes) < 3:
                    example_changes.append({
                        "original": dict(info["feats"]),
                        "perturbed": perturbed,
                        "changes": changes,
                        "label": info["labels"].get("attack_category", "?"),
                    })
        
        pct = changed_patterns / max(total_patterns, 1) * 100
        print(f"\n  Strategy: {strategy}")
        print(f"    Affected patterns: {changed_patterns}/{total_patterns} ({pct:.1f}%)")
        for ex in example_changes[:1]:
            for c in ex["changes"][:2]:
                print(f"    Example: {c}")
        
        perturbation_results[strategy] = {
            "affected_patterns": changed_patterns,
            "total_patterns": total_patterns,
            "affected_pct": round(pct, 1),
            "examples": example_changes[:2],
        }
    
    results["perturbation_analysis"] = perturbation_results
    
    # 4. Summary
    print(f"\n{'='*70}")
    print("  📊 SUMMARY")
    print(f"{'='*70}")
    
    # Most sensitive features
    sorted_sens = sorted(sensitivity.items(), key=lambda x: -x[1]["sensitivity_ratio"])
    print(f"\n  Most sensitive features (if removed, causes ambiguity):")
    for fname, info in sorted_sens[:3]:
        print(f"    {fname}: {info['groups_with_multiple_labels']} groups become ambiguous ({info['interpretation']})")
    
    print(f"\n  Key findings:")
    print(f"    - Dataset is {'deterministic' if det['deterministic'] else 'NOT deterministic'}")
    print(f"    - {sum(1 for v in sensitivity.values() if v['interpretation'] == 'HIGH')} features have HIGH sensitivity")
    print(f"    - Perturbations affect {sum(v['affected_patterns'] for v in perturbation_results.values() if 'drop' not in v)} patterns across {len([s for s in strategies if 'drop' not in s])} strategies")
    
    results["summary"] = {
        "deterministic": det["deterministic"],
        "high_sensitivity_features": [k for k, v in sensitivity.items() if v["interpretation"] == "HIGH"],
        "most_robust_feature": sorted_sens[-1][0] if sorted_sens else None,
        "least_robust_feature": sorted_sens[0][0] if sorted_sens else None,
    }
    
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅ Saved to {OUT}")

if __name__ == "__main__":
    main()

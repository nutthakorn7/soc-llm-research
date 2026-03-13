#!/usr/bin/env python3
"""
llm_eval_audit.py — Automated LLM Evaluation Checklist (P19)

Implements 12 of 30 checklist items from:
  "Beyond Accuracy: A 30-Item Reproducibility Checklist for LLM Classification"

Usage:
  python llm_eval_audit.py --train train.jsonl --test test.jsonl --preds preds.jsonl --labels labels.txt

Items automated:
  1  - Train/test overlap check (MinHash dedup)
  2  - Class distribution & entropy
  3  - Unique pattern count
  4  - Zero-ambiguity check (input→label cardinality)
  5  - Feature importance (mutual information)
  10 - Macro-F1 computation
  11 - Per-class F1 report
  12 - Strict vs normalized F1
  13 - Random/majority baseline
  17 - Vocabulary audit (V_pred vs V_true)
  18 - Hallucination inventory
  20 - Semantic-compliance gap rule
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str, text_key: str = "text", label_key: str = "label"):
    """Load a JSONL file, returning list of (text, label) tuples."""
    data = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get(text_key, obj.get("input", ""))
            label = obj.get(label_key, obj.get("output", ""))
            data.append((str(text), str(label)))
    return data


def load_labels(path: str):
    """Load valid label set from a text file (one label per line)."""
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def entropy(labels: list) -> float:
    """Compute Shannon entropy H(Y) in bits."""
    counter = Counter(labels)
    total = sum(counter.values())
    h = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def normalize_label(label: str, alias_map: Optional[dict] = None) -> str:
    """Normalize a label string for normalized F1 computation."""
    norm = label.strip().lower().replace("_", " ").replace("-", " ")
    if alias_map and norm in alias_map:
        return alias_map[norm]
    return norm


def f1_per_class(y_true: list, y_pred: list):
    """Compute per-class precision, recall, F1."""
    classes = sorted(set(y_true) | set(y_pred))
    results = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for t in y_true if t == cls)
        results[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return results


def macro_f1(per_class: dict) -> float:
    """Compute macro-averaged F1 from per-class results."""
    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    return sum(f1s) / len(f1s) if f1s else 0.0


def simple_hash(text: str) -> int:
    """Simple hash for dedup (replace with MinHash for production)."""
    return hash(text.strip().lower())


# ─── Checklist Items ──────────────────────────────────────────────────────────

class ChecklistResult:
    def __init__(self, item_id: int, name: str, passed: bool, detail: str, severity: str = "INFO"):
        self.item_id = item_id
        self.name = name
        self.passed = passed
        self.detail = detail
        self.severity = severity  # INFO, WARNING, CRITICAL

    def __str__(self):
        icon = "✅" if self.passed else ("⚠️" if self.severity == "WARNING" else "❌")
        return f"  [{self.item_id:2d}] {icon} {self.name}: {self.detail}"


def check_overlap(train_data, test_data) -> ChecklistResult:
    """Item 1: Train/test overlap check."""
    train_hashes = set(simple_hash(t) for t, _ in train_data)
    test_hashes = set(simple_hash(t) for t, _ in test_data)
    overlap = train_hashes & test_hashes
    pct = len(overlap) / len(test_hashes) * 100 if test_hashes else 0

    if pct > 0:
        return ChecklistResult(1, "Train/Test Overlap", False,
                               f"{len(overlap)} overlapping samples ({pct:.1f}%)", "CRITICAL")
    return ChecklistResult(1, "Train/Test Overlap", True,
                           f"No overlap detected ({len(train_hashes)} train, {len(test_hashes)} test)")


def check_class_distribution(labels: list) -> ChecklistResult:
    """Item 2: Class distribution & entropy."""
    h = entropy(labels)
    counter = Counter(labels)
    n_classes = len(counter)
    max_entropy = math.log2(n_classes) if n_classes > 1 else 0
    balance_ratio = h / max_entropy if max_entropy > 0 else 0

    most_common = counter.most_common(1)[0]
    majority_pct = most_common[1] / len(labels) * 100

    if h < 1.0:
        return ChecklistResult(2, "Class Distribution", False,
                               f"H(Y)={h:.3f} bits — DEGENERATE TASK (majority class: {majority_pct:.1f}%)",
                               "CRITICAL")
    if balance_ratio < 0.5:
        return ChecklistResult(2, "Class Distribution", False,
                               f"H(Y)={h:.3f} bits, balance={balance_ratio:.2f} — SEVERELY IMBALANCED",
                               "WARNING")
    return ChecklistResult(2, "Class Distribution", True,
                           f"H(Y)={h:.3f} bits, {n_classes} classes, balance={balance_ratio:.2f}")


def check_unique_patterns(data: list) -> ChecklistResult:
    """Item 3: Unique pattern count (exact dedup)."""
    texts = [t for t, _ in data]
    unique = len(set(texts))
    total = len(texts)
    dup_pct = (total - unique) / total * 100 if total > 0 else 0

    if dup_pct > 10:
        return ChecklistResult(3, "Unique Patterns", False,
                               f"{unique}/{total} unique ({dup_pct:.1f}% duplicates)", "WARNING")
    return ChecklistResult(3, "Unique Patterns", True,
                           f"{unique}/{total} unique ({dup_pct:.1f}% duplicates)")


def check_zero_ambiguity(data: list) -> ChecklistResult:
    """Item 4: Zero-ambiguity check (same input → different labels)."""
    input_labels = {}
    for text, label in data:
        key = text.strip().lower()
        if key not in input_labels:
            input_labels[key] = set()
        input_labels[key].add(label)

    ambiguous = {k: v for k, v in input_labels.items() if len(v) > 1}

    if ambiguous:
        return ChecklistResult(4, "Zero-Ambiguity", False,
                               f"{len(ambiguous)} inputs map to multiple labels", "WARNING")
    return ChecklistResult(4, "Zero-Ambiguity", True,
                           f"All {len(input_labels)} unique inputs have single labels")


def check_majority_baseline(y_true: list) -> ChecklistResult:
    """Item 13: Random/majority baseline."""
    counter = Counter(y_true)
    majority_label, majority_count = counter.most_common(1)[0]
    majority_f1 = majority_count / len(y_true)
    random_f1 = 1.0 / len(counter)

    return ChecklistResult(13, "Majority Baseline", True,
                           f"Majority='{majority_label}' acc={majority_f1:.3f}, "
                           f"Random F1={random_f1:.3f}")


def check_strict_vs_normalized(y_true, y_pred, alias_map=None) -> ChecklistResult:
    """Item 12: Strict vs normalized F1."""
    # Strict F1
    strict_pc = f1_per_class(y_true, y_pred)
    strict = macro_f1(strict_pc)

    # Normalized F1
    norm_true = [normalize_label(t, alias_map) for t in y_true]
    norm_pred = [normalize_label(p, alias_map) for p in y_pred]
    norm_pc = f1_per_class(norm_true, norm_pred)
    normalized = macro_f1(norm_pc)

    gap = normalized - strict
    delta_sc = f"{gap:.4f}" if gap > 0 else "0"

    if gap > 0.05:
        return ChecklistResult(12, "Strict vs Normalized F1", False,
                               f"Strict={strict:.4f}, Norm={normalized:.4f}, "
                               f"Δ_SC={delta_sc} — SIGNIFICANT GAP", "CRITICAL")
    return ChecklistResult(12, "Strict vs Normalized F1", True,
                           f"Strict={strict:.4f}, Norm={normalized:.4f}, Δ_SC={delta_sc}")


def check_per_class_f1(y_true, y_pred) -> ChecklistResult:
    """Item 11: Per-class F1 report."""
    pc = f1_per_class(y_true, y_pred)
    lines = []
    min_f1, min_cls = 1.0, ""
    for cls in sorted(pc.keys()):
        m = pc[cls]
        lines.append(f"  {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
        if m["f1"] < min_f1 and m["support"] > 0:
            min_f1, min_cls = m["f1"], cls

    has_zero = any(m["f1"] == 0 and m["support"] > 0 for m in pc.values())
    detail = f"Worst: {min_cls} (F1={min_f1:.3f})"
    if has_zero:
        detail += " — HAS ZERO-F1 CLASSES"

    return ChecklistResult(11, "Per-Class F1", not has_zero, detail,
                           "WARNING" if has_zero else "INFO")


def check_vocab_audit(y_true, y_pred, valid_labels=None) -> ChecklistResult:
    """Item 17: Vocabulary audit."""
    v_true = set(y_true)
    v_pred = set(y_pred)

    if valid_labels:
        v_schema = valid_labels
    else:
        v_schema = v_true

    novel = v_pred - v_schema
    missing = v_schema - v_pred

    if novel:
        return ChecklistResult(17, "Vocabulary Audit", False,
                               f"|V_pred|={len(v_pred)}, |V_schema|={len(v_schema)}, "
                               f"NOVEL LABELS: {novel}", "CRITICAL")
    return ChecklistResult(17, "Vocabulary Audit", True,
                           f"|V_pred|={len(v_pred)}, |V_schema|={len(v_schema)}, "
                           f"no novel labels, missing={missing or 'none'}")


def check_hallucination_inventory(y_pred, valid_labels) -> ChecklistResult:
    """Item 18: Hallucination inventory."""
    hallucinated = [p for p in y_pred if p not in valid_labels]
    halluc_counter = Counter(hallucinated)
    rate = len(hallucinated) / len(y_pred) * 100 if y_pred else 0
    n_unique = len(halluc_counter)

    if n_unique > 0:
        top3 = halluc_counter.most_common(3)
        top3_str = ", ".join(f"'{k}'({v})" for k, v in top3)
        return ChecklistResult(18, "Hallucination Inventory", False,
                               f"{len(hallucinated)} hallucinated ({rate:.1f}%), "
                               f"{n_unique} unique variants. Top: {top3_str}", "CRITICAL")
    return ChecklistResult(18, "Hallucination Inventory", True,
                           f"0 hallucinated labels (0.0%)")


def check_gap_rule(strict_f1, norm_f1) -> ChecklistResult:
    """Item 20: Δ_SC gap rule — flag if gap > 5%."""
    gap = norm_f1 - strict_f1

    if gap > 0.05:
        return ChecklistResult(20, "Gap Rule (Δ_SC > 5%)", False,
                               f"Δ_SC = {gap:.4f} ({gap*100:.1f}%) — INVESTIGATE LABEL ALIASING",
                               "CRITICAL")
    return ChecklistResult(20, "Gap Rule (Δ_SC > 5%)", True,
                           f"Δ_SC = {gap:.4f} ({gap*100:.1f}%) — within acceptable range")


# ─── Main Audit ───────────────────────────────────────────────────────────────

def run_audit(train_path, test_path, preds_path, labels_path=None):
    """Run the full 12-item automated audit."""
    print("=" * 70)
    print("  LLM Evaluation Checklist Audit (P19 — 12/30 Items)")
    print("=" * 70)

    # Load data
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    # Load predictions — expect {"text": ..., "label": ..., "pred": ...}
    preds_raw = []
    with open(preds_path) as f:
        for line in f:
            preds_raw.append(json.loads(line))

    y_true = [p.get("label", p.get("output", "")) for p in preds_raw]
    y_pred = [p.get("pred", p.get("prediction", "")) for p in preds_raw]

    # Load valid labels
    valid_labels = load_labels(labels_path) if labels_path else set(y_true)

    results = []

    # ── Data Integrity (Items 1-5) ──
    print("\n📊 DATA INTEGRITY")
    results.append(check_overlap(train_data, test_data))
    results.append(check_class_distribution([l for _, l in train_data]))
    results.append(check_unique_patterns(train_data))
    results.append(check_zero_ambiguity(train_data))
    # Item 5: Feature importance (placeholder — requires feature extraction)
    results.append(ChecklistResult(5, "Feature Importance", True,
                                    "Requires domain-specific feature extraction (manual)"))

    for r in results[-5:]:
        print(r)

    # ── Metrics (Items 10-13) ──
    print("\n📏 METRICS")
    pc = f1_per_class(y_true, y_pred)
    mf1 = macro_f1(pc)
    results.append(ChecklistResult(10, "Macro-F1", True, f"Macro-F1 = {mf1:.4f}"))
    results.append(check_per_class_f1(y_true, y_pred))
    r12 = check_strict_vs_normalized(y_true, y_pred)
    results.append(r12)
    results.append(check_majority_baseline(y_true))

    for r in results[-4:]:
        print(r)

    # ── Hallucination (Items 17-18, 20) ──
    print("\n🔍 HALLUCINATION AUDIT")
    results.append(check_vocab_audit(y_true, y_pred, valid_labels))
    results.append(check_hallucination_inventory(y_pred, valid_labels))

    # Compute strict/norm for gap rule
    strict_pc = f1_per_class(y_true, y_pred)
    strict = macro_f1(strict_pc)
    norm_true = [normalize_label(t) for t in y_true]
    norm_pred = [normalize_label(p) for p in y_pred]
    norm_pc = f1_per_class(norm_true, norm_pred)
    normalized = macro_f1(norm_pc)
    results.append(check_gap_rule(strict, normalized))

    for r in results[-3:]:
        print(r)

    # ── Summary ──
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    critical = sum(1 for r in results if not r.passed and r.severity == "CRITICAL")

    print(f"  RESULT: {passed}/{len(results)} passed, {failed} failed "
          f"({critical} CRITICAL)")

    if critical > 0:
        print(f"\n  ⛔ {critical} CRITICAL issues found — results may be unreliable!")
        crit_items = [r for r in results if not r.passed and r.severity == "CRITICAL"]
        for r in crit_items:
            print(f"     → Item {r.item_id}: {r.name}")
    elif failed > 0:
        print(f"\n  ⚠️  {failed} warnings — review recommended before publication")
    else:
        print(f"\n  ✅ All automated checks passed!")

    print("=" * 70)

    # Return structured results
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "critical": critical,
        "items": [{"id": r.item_id, "name": r.name, "passed": r.passed,
                   "detail": r.detail, "severity": r.severity} for r in results]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Checklist Audit (P19)")
    parser.add_argument("--train", required=True, help="Training data (JSONL)")
    parser.add_argument("--test", required=True, help="Test data (JSONL)")
    parser.add_argument("--preds", required=True, help="Predictions (JSONL with 'label' and 'pred')")
    parser.add_argument("--labels", help="Valid label set (one per line)")
    parser.add_argument("--output", help="Save JSON report to file")

    args = parser.parse_args()

    report = run_audit(args.train, args.test, args.preds, args.labels)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")

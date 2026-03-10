#!/usr/bin/env python3
"""
Evaluate fine-tuned SOC Alert Classification model.
Compares against TRUST-SOC ICL baselines.
"""
import json
import os
import re
import argparse
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# TRUST-SOC Baseline Results (Macro-F1 averages)
TRUST_SOC_BASELINES = {
    "Claude Opus 4.6 (5S)":     {"f1": 0.907, "cost": 672},
    "Claude Opus 4.6 (AVG)":    {"f1": 0.815, "cost": 672},
    "Kimi K2 (AVG)":            {"f1": 0.729, "cost": 0},
    "Llama 4 Maverick (AVG)":   {"f1": 0.721, "cost": 7},
    "DeepSeek V3.2 (AVG)":      {"f1": 0.709, "cost": 9},
    "Grok 4.1 Fast (AVG)":      {"f1": 0.603, "cost": 5},
    "GPT-5.2 (AVG)":            {"f1": 0.602, "cost": 75},
}


def parse_response(response_text):
    """Parse model response to extract classification fields."""
    result = {
        "classification": None,
        "triage": None,
        "attack_category": None,
        "priority_score": None
    }

    lines = response_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("classification:"):
            val = line.split(":", 1)[1].strip()
            result["classification"] = val
        elif line.lower().startswith("triage decision:") or line.lower().startswith("triage:"):
            val = line.split(":", 1)[1].strip()
            result["triage"] = val
        elif line.lower().startswith("attack category:"):
            val = line.split(":", 1)[1].strip()
            result["attack_category"] = val
        elif line.lower().startswith("priority score:"):
            val = line.split(":", 1)[1].strip()
            try:
                result["priority_score"] = float(val)
            except ValueError:
                pass

    return result


def evaluate_predictions(test_data, predictions):
    """Calculate metrics for all tasks."""
    results = {}

    # Task 1: Binary Classification (Malicious/Benign)
    true_cls = []
    pred_cls = []
    for td, pred in zip(test_data, predictions):
        true_resp = parse_response(td["conversations"][-1]["value"])
        pred_resp = parse_response(pred)
        if true_resp["classification"] and pred_resp["classification"]:
            true_cls.append(true_resp["classification"].lower())
            pred_cls.append(pred_resp["classification"].lower())

    if true_cls:
        results["binary_classification"] = {
            "macro_f1": f1_score(true_cls, pred_cls, average="macro"),
            "precision": precision_score(true_cls, pred_cls, average="macro"),
            "recall": recall_score(true_cls, pred_cls, average="macro"),
            "n_samples": len(true_cls),
            "report": classification_report(true_cls, pred_cls)
        }

    # Task 2: Triage (3-class)
    true_tri = []
    pred_tri = []
    for td, pred in zip(test_data, predictions):
        true_resp = parse_response(td["conversations"][-1]["value"])
        pred_resp = parse_response(pred)
        if true_resp["triage"] and pred_resp["triage"]:
            true_tri.append(true_resp["triage"].lower())
            pred_tri.append(pred_resp["triage"].lower())

    if true_tri:
        results["triage"] = {
            "macro_f1": f1_score(true_tri, pred_tri, average="macro"),
            "precision": precision_score(true_tri, pred_tri, average="macro"),
            "recall": recall_score(true_tri, pred_tri, average="macro"),
            "n_samples": len(true_tri),
            "report": classification_report(true_tri, pred_tri)
        }

    # Task 3: Attack Category (15-class)
    true_atk = []
    pred_atk = []
    for td, pred in zip(test_data, predictions):
        true_resp = parse_response(td["conversations"][-1]["value"])
        pred_resp = parse_response(pred)
        if true_resp["attack_category"] and pred_resp["attack_category"]:
            true_atk.append(true_resp["attack_category"].lower())
            pred_atk.append(pred_resp["attack_category"].lower())

    if true_atk:
        results["attack_category"] = {
            "macro_f1": f1_score(true_atk, pred_atk, average="macro"),
            "precision": precision_score(true_atk, pred_atk, average="macro"),
            "recall": recall_score(true_atk, pred_atk, average="macro"),
            "n_samples": len(true_atk),
            "report": classification_report(true_atk, pred_atk, zero_division=0)
        }

    return results


def print_comparison(results):
    """Print comparison table with TRUST-SOC baselines."""
    # Use binary classification F1 as main metric
    our_f1 = results.get("binary_classification", {}).get("macro_f1", 0)
    triage_f1 = results.get("triage", {}).get("macro_f1", 0)
    attack_f1 = results.get("attack_category", {}).get("macro_f1", 0)

    print("\n" + "=" * 70)
    print("COMPARISON WITH TRUST-SOC BASELINES")
    print("=" * 70)
    print(f"{'Model':<30} {'Method':<15} {'Cost':>8} {'Macro-F1':>10}")
    print("-" * 70)

    # Add our result
    print(f"{'** Llama 3.1 8B (Ours) **':<30} {'Fine-tuned':<15} {'$0':>8} {our_f1:>10.1%}")
    print("-" * 70)

    # Add baselines
    for name, data in TRUST_SOC_BASELINES.items():
        cost_str = f"${data['cost']}"
        print(f"{name:<30} {'ICL':<15} {cost_str:>8} {data['f1']:>10.1%}")

    print("=" * 70)

    print(f"\n--- Per-Task Results ---")
    print(f"  Binary Classification: F1 = {our_f1:.4f}")
    print(f"  Triage (3-class):      F1 = {triage_f1:.4f}")
    print(f"  Attack Category:       F1 = {attack_f1:.4f}")

    # Highlight winner
    best_baseline_f1 = max(d["f1"] for d in TRUST_SOC_BASELINES.values())
    if our_f1 > best_baseline_f1:
        print(f"\n🏆 Fine-tuned model WINS against all baselines! ({our_f1:.1%} > {best_baseline_f1:.1%})")
    else:
        beaten = [n for n, d in TRUST_SOC_BASELINES.items() if our_f1 > d["f1"]]
        if beaten:
            print(f"\n✅ Fine-tuned model beats: {', '.join(beaten)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True, help="Path to test JSON (LlamaFactory format)")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--output", default="results.json", help="Output results JSON")
    args = parser.parse_args()

    # Load data
    with open(args.test_data) as f:
        test_data = json.load(f)
    with open(args.predictions) as f:
        predictions = json.load(f)

    print(f"Test samples: {len(test_data)}")
    print(f"Predictions:  {len(predictions)}")

    # Evaluate
    results = evaluate_predictions(test_data, predictions)

    # Print comparison
    print_comparison(results)

    # Save results
    # Convert to JSON-serializable
    save_results = {}
    for task, metrics in results.items():
        save_results[task] = {k: v for k, v in metrics.items() if k != "report"}

    with open(args.output, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

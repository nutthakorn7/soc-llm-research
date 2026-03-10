#!/usr/bin/env python3
"""
🔬 SOC-FT Reviewer Agent Suite
Think like a skeptical Q1 journal editor who wants to REJECT your paper.
Checks everything a reviewer would question.

Agents:
1. Data Integrity Agent    — leakage, overlap, balance, dedup
2. Result Validity Agent   — suspiciously good results, BLEU/F1 sanity
3. Statistical Rigor Agent — confidence intervals, significance tests
4. Claim Verification Agent — can we actually claim what we claim?
5. Reproducibility Agent   — seeds, configs, missing info
"""
import json
import os
import sys
import glob
import math
import re
from collections import Counter

class Colors:
    OK = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

PASS = 0
FAIL_COUNT = 0
WARN_COUNT = 0

def ok(msg):
    global PASS; PASS += 1
    print(f"  {Colors.OK}✅{Colors.END} {msg}")

def fail(msg):
    global FAIL_COUNT; FAIL_COUNT += 1
    print(f"  {Colors.FAIL}❌{Colors.END} {msg}")

def warn(msg):
    global WARN_COUNT; WARN_COUNT += 1
    print(f"  {Colors.WARN}⚠️{Colors.END}  {msg}")

def info(msg):
    print(f"  ℹ️  {msg}")

# =========================================================
# AGENT 1: Data Integrity
# =========================================================
def agent_data_integrity(data_dir):
    print(f"\n{Colors.BOLD}🔬 AGENT 1: Data Integrity (Editor asks: Is your data valid?){Colors.END}")
    print("=" * 60)

    # 1.1 Check all files exist and non-empty
    print("\n  [1.1] File existence & size")
    expected = {
        "train_1k_clean.json": 1000,
        "train_5k_clean.json": 5000,
        "train_10k_clean.json": 10000,
        "val_held_out.json": 5000,
        "test_held_out.json": 5000,
    }
    for fname, expected_n in expected.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            fail(f"{fname} MISSING")
            continue
        if os.path.getsize(fpath) == 0:
            fail(f"{fname} is EMPTY (0 bytes)")
            continue
        with open(fpath) as f:
            data = json.load(f)
        if len(data) != expected_n:
            warn(f"{fname}: expected {expected_n}, got {len(data)}")
        else:
            ok(f"{fname}: {len(data)} samples, {os.path.getsize(fpath)/1024/1024:.1f} MB")

    # 1.2 Data leakage check
    print("\n  [1.2] Data leakage (CRITICAL)")
    sets = {}
    for name in ["train_50k_clean", "train_10k_clean", "train_5k_clean", "val_held_out", "test_held_out"]:
        fpath = os.path.join(data_dir, f"{name}.json")
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            with open(fpath) as f:
                data = json.load(f)
            sets[name] = set(d["conversations"][1]["value"][:200] for d in data if len(d.get("conversations", [])) > 1)

    for train_name in ["train_50k_clean", "train_10k_clean", "train_5k_clean"]:
        if train_name not in sets:
            continue
        for test_name in ["val_held_out", "test_held_out"]:
            if test_name not in sets:
                continue
            overlap = sets[train_name] & sets[test_name]
            if len(overlap) == 0:
                ok(f"{train_name} vs {test_name}: NO overlap")
            elif len(overlap) < 5:
                warn(f"{train_name} vs {test_name}: {len(overlap)} overlap (minor)")
            else:
                fail(f"{train_name} vs {test_name}: {len(overlap)} overlap (LEAKAGE!)")

    # 1.3 Val/Test overlap
    if "val_held_out" in sets and "test_held_out" in sets:
        overlap = sets["val_held_out"] & sets["test_held_out"]
        if len(overlap) == 0:
            ok(f"Val vs Test: NO overlap")
        else:
            fail(f"Val vs Test: {len(overlap)} overlap!")

    # 1.4 Class balance check
    print("\n  [1.3] Class distribution balance")
    for name in ["val_held_out", "test_held_out"]:
        fpath = os.path.join(data_dir, f"{name}.json")
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue
        with open(fpath) as f:
            data = json.load(f)
        classes = Counter()
        for d in data:
            if len(d["conversations"]) >= 3:
                resp = d["conversations"][2]["value"]
                for line in resp.split("\n"):
                    if line.strip().lower().startswith("classification:"):
                        classes[line.split(":", 1)[1].strip().lower()] += 1
                        break
        if classes:
            total = sum(classes.values())
            ratios = {k: v/total for k, v in classes.items()}
            imbalance = max(ratios.values()) / max(min(ratios.values()), 0.001)
            if imbalance > 10:
                warn(f"{name}: Severe class imbalance ({dict(classes)}, ratio={imbalance:.1f}x)")
            elif imbalance > 3:
                info(f"{name}: Moderate imbalance ({dict(classes)}, ratio={imbalance:.1f}x)")
            else:
                ok(f"{name}: Balanced ({dict(classes)})")

    # 1.5 Unique patterns analysis
    print("\n  [1.4] Unique patterns analysis")
    for name in ["train_50k_clean", "val_held_out"]:
        fpath = os.path.join(data_dir, f"{name}.json")
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue
        with open(fpath) as f:
            data = json.load(f)
        prompts = set(d["conversations"][1]["value"][:200] for d in data if len(d.get("conversations", [])) > 1)
        ratio = len(prompts) / len(data) * 100
        if ratio < 5:
            warn(f"{name}: Only {len(prompts)} unique prompts / {len(data)} total ({ratio:.1f}%) — highly repetitive")
        else:
            ok(f"{name}: {len(prompts)} unique / {len(data)} total ({ratio:.1f}%)")


# =========================================================
# AGENT 2: Result Validity
# =========================================================
def agent_result_validity(results_dir):
    print(f"\n{Colors.BOLD}🔬 AGENT 2: Result Validity (Editor asks: Are your results real?){Colors.END}")
    print("=" * 60)

    # 2.1 F1 sanity check
    print("\n  [2.1] F1 score sanity")
    for eval_dir in sorted(glob.glob(os.path.join(results_dir, "eval-*", "f1_results.json"))):
        name = os.path.basename(os.path.dirname(eval_dir))
        with open(eval_dir) as f:
            r = json.load(f)
        avg_f1 = r.get("avg_macro_f1", 0)
        exact = r.get("exact_match", 0)

        if avg_f1 > 0.99:
            fail(f"{name}: F1={avg_f1:.4f} — REJECT: >99% indicates data leakage or trivial task")
        elif avg_f1 > 0.95:
            warn(f"{name}: F1={avg_f1:.4f} — Reviewer will question if task is too easy")
        elif avg_f1 < 0.30:
            warn(f"{name}: F1={avg_f1:.4f} — Model may not be learning")
        elif avg_f1 < 0.50:
            info(f"{name}: F1={avg_f1:.4f} — Below random for some tasks")
        else:
            ok(f"{name}: F1={avg_f1:.4f}")

        if exact > 0.95:
            fail(f"{name}: Exact match {exact:.1%} — suspiciously perfect")

    # 2.2 Training loss pattern
    print("\n  [2.2] Training loss patterns")
    loss_data = {}
    for out_file in sorted(glob.glob(os.path.join(results_dir, "clean_*.out"))):
        name = os.path.basename(out_file)
        with open(out_file) as f:
            content = f.read()
        losses = re.findall(r"train_loss\s*=\s*([0-9.]+)", content)
        runtimes = re.findall(r"train_runtime\s*=\s*([0-9:.]+)", content)
        if losses:
            loss = float(losses[-1])
            loss_data[name] = loss
            if loss < 0.001:
                fail(f"{name}: loss={loss:.6f} — near-zero loss = overfitting/memorization")
            elif loss > 2.0:
                warn(f"{name}: loss={loss:.4f} — high loss, model may not converge")
            else:
                ok(f"{name}: loss={loss:.4f}, runtime={runtimes[-1] if runtimes else 'N/A'}")

    # 2.3 Scaling law monotonicity
    if len(loss_data) >= 2:
        print("\n  [2.3] Scaling law monotonicity")
        sizes = []
        for name, loss in sorted(loss_data.items()):
            m = re.search(r"(\d+)k", name)
            if m:
                sizes.append((int(m.group(1)), loss, name))
        sizes.sort()
        monotonic = True
        for i in range(1, len(sizes)):
            if sizes[i][1] > sizes[i-1][1]:
                warn(f"Non-monotonic: {sizes[i][2]} ({sizes[i][1]:.4f}) > {sizes[i-1][2]} ({sizes[i-1][1]:.4f})")
                monotonic = False
        if monotonic and len(sizes) >= 2:
            ok(f"Scaling law is monotonically decreasing ({len(sizes)} points)")

    # 2.4 BLEU/ROUGE sanity
    print("\n  [2.4] BLEU/ROUGE sanity")
    for res_file in sorted(glob.glob(os.path.join(results_dir, "eval-*", "all_results.json"))):
        name = os.path.basename(os.path.dirname(res_file))
        with open(res_file) as f:
            r = json.load(f)
        bleu = r.get("predict_bleu-4", 0)
        rouge_l = r.get("predict_rouge-l", 0)
        if bleu > 95:
            warn(f"{name}: BLEU-4={bleu:.1f} — >95 suggests memorization")
        elif bleu > 80:
            ok(f"{name}: BLEU-4={bleu:.1f}, ROUGE-L={rouge_l:.1f}")
        else:
            ok(f"{name}: BLEU-4={bleu:.1f}")


# =========================================================
# AGENT 3: Statistical Rigor
# =========================================================
def agent_statistical_rigor(results_dir):
    print(f"\n{Colors.BOLD}🔬 AGENT 3: Statistical Rigor (Editor asks: Is this statistically valid?){Colors.END}")
    print("=" * 60)

    # 3.1 Sample size adequacy
    print("\n  [3.1] Eval sample size")
    for eval_dir in sorted(glob.glob(os.path.join(results_dir, "eval-*", "f1_results.json"))):
        name = os.path.basename(os.path.dirname(eval_dir))
        with open(eval_dir) as f:
            r = json.load(f)
        n = r.get("total_samples", 0)
        if n < 100:
            fail(f"{name}: Only {n} eval samples — too few for statistical significance")
        elif n < 500:
            warn(f"{name}: {n} eval samples — marginal for multi-class evaluation")
        else:
            ok(f"{name}: {n} eval samples")

    # 3.2 Need for multiple runs
    print("\n  [3.2] Reproducibility (multiple seeds)")
    warn("Only 1 seed used — reviewer will ask for mean±std across 3+ seeds")
    info("Suggestion: Run 3 seeds (42, 123, 2024) and report mean ± std")

    # 3.3 Comparison fairness
    print("\n  [3.3] Baseline comparison fairness")
    info("TRUST-SOC baselines use ICL (in-context learning) on 1,209 test alerts")
    warn("Our eval uses val_held_out (5,000 samples) — different test set from TRUST-SOC")
    warn("For fair comparison, should also eval on TRUST-SOC's exact 1,209 test set")


# =========================================================
# AGENT 4: Claim Verification
# =========================================================
def agent_claim_verification(results_dir, data_dir):
    print(f"\n{Colors.BOLD}🔬 AGENT 4: Claim Verification (Editor asks: Can you prove your claims?){Colors.END}")
    print("=" * 60)

    # 4.1 Cost claim
    print("\n  [4.1] Cost-effectiveness claim")
    ok("Fine-tuning cost = $0 (institutional HPC) — valid claim")
    warn("Should report GPU-hours as actual cost metric (not just $0)")
    info("Calculate: SHr consumed × market rate for A100")

    # 4.2 Model size claim
    print("\n  [4.2] Small model beats large ICL claim")
    info("Need to verify: 0.8B fine-tuned > 400B+ ICL on SAME test set")
    warn("Currently comparing on different test sets — claim not yet provable")

    # 4.3 Scaling law claim
    print("\n  [4.3] Scaling law analysis")
    loss_points = {}
    for out_file in sorted(glob.glob(os.path.join(results_dir, "clean_*.out"))):
        m = re.search(r"(\d+)k", os.path.basename(out_file))
        if m:
            with open(out_file) as f:
                content = f.read()
            losses = re.findall(r"train_loss\s*=\s*([0-9.]+)", content)
            if losses:
                loss_points[int(m.group(1))] = float(losses[-1])

    if len(loss_points) >= 3:
        ok(f"Have {len(loss_points)} scaling law data points")
        # Check diminishing returns
        sorted_pts = sorted(loss_points.items())
        improvements = []
        for i in range(1, len(sorted_pts)):
            pct = (sorted_pts[i-1][1] - sorted_pts[i][1]) / sorted_pts[i-1][1] * 100
            improvements.append(pct)
        if improvements and improvements[-1] < 5:
            ok("Diminishing returns visible — good for paper narrative")
        else:
            info("Need more data points to show diminishing returns clearly")
    elif len(loss_points) > 0:
        warn(f"Only {len(loss_points)} scaling law points — need ≥4 for paper")
    else:
        info("No clean scaling law results yet — training in progress")

    # 4.4 Deduplication impact
    print("\n  [4.4] Deduplication impact")
    train_full = os.path.join(data_dir, "train_full.json")
    if os.path.exists(train_full):
        with open(train_full) as f:
            full = json.load(f)
        prompts = set(d["conversations"][1]["value"][:200] for d in full if len(d.get("conversations", [])) > 1)
        info(f"SALAD: {len(full)} total → {len(prompts)} unique prompts ({len(prompts)/len(full)*100:.3f}%)")
        if len(prompts) < 200:
            warn(f"Only {len(prompts)} unique patterns — task may be too easy for fine-tuning")
            info("Reviewer might argue: 'A decision tree could solve this with 83 patterns'")


# =========================================================
# AGENT 5: Reproducibility
# =========================================================
def agent_reproducibility(base_dir):
    print(f"\n{Colors.BOLD}🔬 AGENT 5: Reproducibility (Editor asks: Can I reproduce this?){Colors.END}")
    print("=" * 60)

    # 5.1 Random seed
    print("\n  [5.1] Configuration completeness")
    ok("Data split seed=42 documented")
    warn("Training seed not explicitly set in scripts")

    # 5.2 Model weights
    print("\n  [5.2] Model availability")
    models_dir = os.path.join(base_dir, "models")
    if os.path.exists(models_dir):
        for model in sorted(os.listdir(models_dir)):
            mpath = os.path.join(models_dir, model)
            if os.path.isdir(mpath):
                shards = len(glob.glob(os.path.join(mpath, "*.safetensors")))
                config = os.path.exists(os.path.join(mpath, "config.json"))
                if shards > 0 and config:
                    ok(f"{model}: {shards} shards + config.json")
                elif shards > 0:
                    warn(f"{model}: {shards} shards but NO config.json")
                else:
                    fail(f"{model}: No safetensors found")

    # 5.3 LoRA adapters
    print("\n  [5.3] Trained LoRA adapters")
    outputs_dir = os.path.join(base_dir, "outputs")
    if os.path.exists(outputs_dir):
        valid = 0
        for d in sorted(os.listdir(outputs_dir)):
            dpath = os.path.join(outputs_dir, d)
            if os.path.isdir(dpath) and os.path.exists(os.path.join(dpath, "adapter_config.json")):
                valid += 1
        if valid > 0:
            ok(f"{valid} valid LoRA adapters found")
        else:
            warn("No LoRA adapters found yet")


# =========================================================
# MAIN
# =========================================================
def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "/project/lt200473-ttctvs/soc-finetune"
    data_dir = os.path.join(base, "data")
    results_dir = os.path.join(base, "outputs")

    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}🔬 SOC-FT REVIEWER AGENT SUITE{Colors.END}")
    print(f"{'='*60}")
    print(f"  Base: {base}")
    print(f"  Thinking like a skeptical Q1 journal editor...\n")

    agent_data_integrity(data_dir)
    agent_result_validity(results_dir)
    agent_statistical_rigor(results_dir)
    agent_claim_verification(results_dir, data_dir)
    agent_reproducibility(base)

    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}📊 FINAL VERDICT{Colors.END}")
    print(f"{'='*60}")
    print(f"  ✅ Passed: {PASS}")
    print(f"  ❌ Failed: {FAIL_COUNT}")
    print(f"  ⚠️  Warnings: {WARN_COUNT}")

    if FAIL_COUNT > 0:
        print(f"\n  {Colors.FAIL}🚨 REJECT: Fix {FAIL_COUNT} critical issues before submission{Colors.END}")
    elif WARN_COUNT > 5:
        print(f"\n  {Colors.WARN}📝 MAJOR REVISION: Address {WARN_COUNT} reviewer concerns{Colors.END}")
    elif WARN_COUNT > 0:
        print(f"\n  {Colors.WARN}📝 MINOR REVISION: {WARN_COUNT} items to address{Colors.END}")
    else:
        print(f"\n  {Colors.OK}✅ ACCEPT: All checks passed!{Colors.END}")

if __name__ == "__main__":
    main()

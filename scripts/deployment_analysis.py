#!/usr/bin/env python3
"""
Deployment Scenario Analysis for SOC-FT Paper
Calculates: given X alerts/day, which model + method is optimal?
"""
import json

# From our experiments
METHODS = {
    "Decision Tree": {
        "latency_ms": 0.01,
        "gpu_required": False,
        "accuracy": {"cls": 1.0, "tri": 1.0, "atk": 0.874},
        "cost_per_10k": 0,  # CPU only
        "explanation": False,
        "multi_task": False,
    },
    "Qwen3.5-0.8B FT (QLoRA)": {
        "latency_ms": 500,  # estimate, will update from benchmark
        "gpu_required": True,
        "gpu_mem_gb": 2.5,
        "accuracy": {"cls": None, "tri": None, "atk": None},  # pending
        "cost_per_10k": 0.08,
        "explanation": True,
        "multi_task": True,
    },
    "Qwen3.5-9B FT (QLoRA)": {
        "latency_ms": 3000,  # estimate
        "gpu_required": True,
        "gpu_mem_gb": 8.0,
        "accuracy": {"cls": None, "tri": None, "atk": None},  # pending
        "cost_per_10k": 0.50,
        "explanation": True,
        "multi_task": True,
    },
    "Claude Opus 4.6 ICL": {
        "latency_ms": 2000,
        "gpu_required": False,  # API
        "accuracy": {"cls": None, "tri": None, "atk": None, "avg": 0.907},
        "cost_per_10k": 672,
        "explanation": True,
        "multi_task": True,
    },
    "GPT-5.2 ICL": {
        "latency_ms": 1500,
        "gpu_required": False,
        "accuracy": {"cls": None, "tri": None, "atk": None, "avg": 0.602},
        "cost_per_10k": 75,
        "explanation": True,
        "multi_task": True,
    },
}

SCENARIOS = [
    {"name": "Small SOC (Startup)", "alerts_day": 1_000, "budget_month": 500},
    {"name": "Medium SOC (Enterprise)", "alerts_day": 10_000, "budget_month": 5_000},
    {"name": "Large SOC (MSSP)", "alerts_day": 100_000, "budget_month": 50_000},
    {"name": "Mega SOC (Telco/Gov)", "alerts_day": 1_000_000, "budget_month": 200_000},
]

def analyze():
    print("=" * 80)
    print("  SOC-FT DEPLOYMENT SCENARIO ANALYSIS")
    print("=" * 80)
    
    for scenario in SCENARIOS:
        print(f"\n{'='*80}")
        print(f"  📊 {scenario['name']}")
        print(f"     {scenario['alerts_day']:,} alerts/day | ${scenario['budget_month']:,}/month budget")
        print(f"{'='*80}")
        
        for method, config in METHODS.items():
            alerts_day = scenario["alerts_day"]
            
            # Can it handle the throughput?
            alerts_per_day_capacity = (86400 * 1000) / config["latency_ms"]
            can_handle = alerts_per_day_capacity >= alerts_day
            gpus_needed = max(1, int(alerts_day / alerts_per_day_capacity) + 1) if config["gpu_required"] else 0
            
            # Monthly cost
            if "cost_per_10k" in config:
                monthly_cost = (alerts_day * 30 / 10_000) * config["cost_per_10k"]
            else:
                monthly_cost = 0
            
            if config["gpu_required"]:
                gpu_cost_month = gpus_needed * 2 * 24 * 30  # $2/hr/GPU
                monthly_cost += gpu_cost_month
            
            within_budget = monthly_cost <= scenario["budget_month"]
            
            feasible = "✅" if (can_handle or gpus_needed <= 10) and within_budget else "❌"
            
            print(f"\n  {feasible} {method}")
            print(f"     Capacity: {alerts_per_day_capacity:,.0f} alerts/day (1 GPU)")
            if config["gpu_required"]:
                print(f"     GPUs needed: {gpus_needed} | Mem: {config.get('gpu_mem_gb', '?')} GB/GPU")
            print(f"     Monthly cost: ${monthly_cost:,.0f}")
            print(f"     Explanation: {'✅' if config['explanation'] else '❌'}")
            atk = config['accuracy'].get('atk')
            avg = config['accuracy'].get('avg')
            if atk:
                print(f"     Attack Category F1: {atk:.1%}")
            elif avg:
                print(f"     Avg F1: {avg:.1%}")
    
    # Recommendation matrix
    print(f"\n{'='*80}")
    print("  📋 RECOMMENDATION MATRIX")
    print(f"{'='*80}")
    print(f"  {'Scenario':<25} {'Recommended':<25} {'Why'}")
    print(f"  {'-'*75}")
    print(f"  {'Small (1K/day)':<25} {'DT + 0.8B hybrid':<25} {'DT for cls/tri, LLM for atk'}")
    print(f"  {'Medium (10K/day)':<25} {'0.8B FT (1 GPU)':<25} {'$2.40/day, full NL output'}")
    print(f"  {'Large (100K/day)':<25} {'0.8B FT (2 GPUs)':<25} {'$4.80/day, still cheap'}")
    print(f"  {'Mega (1M/day)':<25} {'DT + 0.8B cascade':<25} {'DT filters, LLM for hard cases'}")

    results = {"scenarios": SCENARIOS, "methods": {k: {kk: vv for kk, vv in v.items() if kk != "accuracy"} for k, v in METHODS.items()}}
    out = "/project/lt200473-ttctvs/soc-finetune/outputs/paper_results/deployment_analysis.json"
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out}")

if __name__ == "__main__":
    analyze()

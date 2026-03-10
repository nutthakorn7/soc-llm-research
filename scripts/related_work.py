#!/usr/bin/env python3
"""
Related Work Comparison Table for SOC-FT Paper.
Positions our work against recent SOC/security LLM papers (2023-2025).
"""

# Recent related work in SOC + LLM space
RELATED_WORK = [
    {
        "paper": "Alam et al. (2024)",
        "title": "Exploring LLM Performance in Cybersecurity Triage",
        "venue": "U. Twente",
        "models": "GPT-4, Llama 3, Mistral, Gemma, Phi-3",
        "method": "ICL (zero/few-shot)",
        "dataset": "Private SOC logs",
        "dataset_size": "~1K",
        "tasks": "Alert classification",
        "multi_model": True,
        "fine_tuning": False,
        "cost_analysis": False,
        "baselines": False,
        "open_data": False,
    },
    {
        "paper": "Simbian (2025)",
        "title": "SOC LLM Benchmark",
        "venue": "Industry",
        "models": "GPT-4, Claude, Gemini, etc.",
        "method": "ICL",
        "dataset": "100 real APT scenarios",
        "dataset_size": "100",
        "tasks": "Alert triage",
        "multi_model": True,
        "fine_tuning": False,
        "cost_analysis": False,
        "baselines": False,
        "open_data": False,
    },
    {
        "paper": "CyberSOCEval (2025)",
        "title": "CrowdStrike + Meta Benchmark",
        "venue": "Industry",
        "models": "Llama, various",
        "method": "ICL + eval",
        "dataset": "CyberSecEval suite",
        "dataset_size": "~5K",
        "tasks": "Investigation, summary, severity",
        "multi_model": True,
        "fine_tuning": False,
        "cost_analysis": False,
        "baselines": False,
        "open_data": True,
    },
    {
        "paper": "CyberLLMInstruct (2025)",
        "title": "Safety-Performance Tradeoff in Fine-tuned LLMs",
        "venue": "arXiv",
        "models": "Llama 3.1 8B",
        "method": "Fine-tuning",
        "dataset": "CyberLLMInstruct",
        "dataset_size": "55K",
        "tasks": "General cybersecurity",
        "multi_model": False,
        "fine_tuning": True,
        "cost_analysis": False,
        "baselines": False,
        "open_data": True,
    },
    {
        "paper": "TRUST-SOC (Ours, 2025)",
        "title": "Benchmarking LLMs for SOC Alert Triage via ICL",
        "venue": "ETRI Journal (submitted)",
        "models": "7 frontier models",
        "method": "ICL (5-shot)",
        "dataset": "SALAD",
        "dataset_size": "1.9M",
        "tasks": "Classification, Triage, Attack Cat, Priority",
        "multi_model": True,
        "fine_tuning": False,
        "cost_analysis": True,
        "baselines": False,
        "open_data": True,
    },
    {
        "paper": "SOC-FT (Ours, 2025)",
        "title": "Task Complexity Analysis: When LLMs Outperform Traditional ML",
        "venue": "Target: Q1",
        "models": "9 models (0.8B-9B)",
        "method": "QLoRA fine-tuning",
        "dataset": "SALAD",
        "dataset_size": "1.9M",
        "tasks": "Classification, Triage, Attack Cat, Priority",
        "multi_model": True,
        "fine_tuning": True,
        "cost_analysis": True,
        "baselines": True,
        "open_data": True,
    },
]

def print_comparison():
    print("=" * 100)
    print("  RELATED WORK COMPARISON")
    print("=" * 100)
    
    # Feature comparison
    features = ["multi_model", "fine_tuning", "cost_analysis", "baselines", "open_data"]
    feature_labels = ["Multi-Model", "Fine-Tuning", "Cost Analysis", "ML Baselines", "Open Data"]
    
    header = f"  {'Paper':<25} {'Dataset':>8} {'Method':>10}"
    for fl in feature_labels:
        header += f" {fl:>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    
    for w in RELATED_WORK:
        row = f"  {w['paper']:<25} {w['dataset_size']:>8} {w['method']:>10}"
        for f in features:
            row += f" {'✅':>12}" if w[f] else f" {'❌':>12}"
        print(row)
    
    print()
    print("  Key Differentiators of SOC-FT:")
    print("  1. ONLY paper with both fine-tuning AND traditional ML baselines")
    print("  2. ONLY paper with task complexity analysis (entropy-based)")
    print("  3. ONLY paper with comprehensive cost analysis ($0.80 vs $5,560)")
    print("  4. Largest SOC alert dataset (1.9M) with open access")
    print("  5. Most models compared (9 architectures, 0.8B-9B)")

if __name__ == "__main__":
    print_comparison()

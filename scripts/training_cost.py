#!/usr/bin/env python3
"""
Training Cost & Carbon Footprint Calculator for SOC-FT Paper

Extracts from training logs:
- GPU hours per model
- Estimated electricity cost
- CO2 footprint
- Cost comparison: FT vs ICL
"""
import json, os, sys, glob
from datetime import datetime, timedelta

BASE = sys.argv[1] if len(sys.argv) > 1 else "/project/lt200473-ttctvs/soc-finetune"
OUT = os.path.join(BASE, "outputs/paper_results/training_cost.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Lanta A100 specs
GPU_TDP_W = 400  # A100 TDP
GPU_COST_PER_HR = 2.0  # Estimated cloud equivalent $/hr
CO2_PER_KWH = 0.5  # kg CO2 per kWh (Thailand grid average)
ELECTRICITY_COST_PER_KWH = 0.12  # USD

# Models and their training dirs
TRAINING_RUNS = {
    # Scaling law (clean)
    "Qwen3.5-9B × 1K": "clean-qwen35-1k",
    "Qwen3.5-9B × 5K": "clean-qwen35-5k",
    "Qwen3.5-9B × 10K": "clean-qwen35-10k",
    "Qwen3.5-9B × 20K": "clean-qwen35-20k",
    # Multi-model
    "DeepSeek-R1 7B × 5K": "mm-dsk-5k",
    "Phi-4-mini 3.8B × 5K": "mm-phi4-5k",
    "Mistral 7B × 5K": "mm-mis-5k",
    "Qwen3-8B × 5K": "mm-q3-5k",
    "Qwen3.5-0.8B × 5K": "mm-q08-5k",
    "SmolLM2 1.7B × 5K": "mm-smol-5k",
    # Seeds
    "Qwen3.5-9B × 5K (seed 123)": "seed-123-q35-5k",
    "Qwen3.5-9B × 5K (seed 2024)": "seed-2024-q35-5k",
    # Ablations
    "Ablation: rank 16": "abl-rank16",
    "Ablation: rank 32": "abl-rank32",
    "Ablation: rank 128": "abl-rank128",
    "Ablation: LR 1e-4": "abl-lr1e-4",
    "Ablation: LR 5e-4": "abl-lr5e-4",
}

# ICL costs (from TRUST-SOC paper)
ICL_COSTS = {
    "GPT-4o (10K alerts)": 556.0,
    "Claude Opus (10K alerts)": 672.0,
    "GPT-4 Turbo (10K alerts)": 340.0,
    "Gemini Pro (10K alerts)": 40.0,
}

def parse_training_time(run_dir):
    """Extract training time from trainer_log.jsonl."""
    log_file = os.path.join(run_dir, "trainer_log.jsonl")
    if not os.path.exists(log_file):
        return None
    
    entries = []
    with open(log_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except:
                pass
    
    if not entries:
        return None
    
    last = entries[-1]
    elapsed = last.get("elapsed_time", "0:00:00")
    total_steps = last.get("total_steps", 0)
    current_steps = last.get("current_steps", 0)
    loss = last.get("loss", None)
    
    # Parse elapsed time
    parts = elapsed.split(":")
    if len(parts) == 3:
        hours = int(parts[0]) + int(parts[1]) / 60 + int(parts[2]) / 3600
    else:
        hours = 0
    
    # If not complete, estimate total time
    if current_steps < total_steps and current_steps > 0:
        hours = hours * (total_steps / current_steps)
    
    return {
        "elapsed_time": elapsed,
        "total_steps": total_steps,
        "current_steps": current_steps,
        "estimated_hours": round(hours, 2),
        "final_loss": loss,
        "completed": current_steps >= total_steps * 0.95,
    }

def main():
    print("=" * 70)
    print("  SOC-FT TRAINING COST ANALYSIS")
    print("=" * 70)
    
    results = []
    total_gpu_hours = 0
    
    print(f"\n  {'Model':<35} {'Hours':>6} {'Steps':>10} {'Loss':>8} {'Cost $':>8} {'CO2 kg':>8}")
    print(f"  {'-'*80}")
    
    for name, dirname in TRAINING_RUNS.items():
        run_dir = os.path.join(BASE, "outputs", dirname)
        info = parse_training_time(run_dir)
        
        if info is None:
            print(f"  {name:<35} {'N/A':>6}")
            continue
        
        hours = info["estimated_hours"]
        total_gpu_hours += hours
        
        # Costs
        electricity_kwh = hours * GPU_TDP_W / 1000
        electricity_cost = electricity_kwh * ELECTRICITY_COST_PER_KWH
        cloud_cost = hours * GPU_COST_PER_HR
        co2_kg = electricity_kwh * CO2_PER_KWH
        
        loss_str = f"{info['final_loss']:.4f}" if info['final_loss'] else "N/A"
        status = "✅" if info["completed"] else "🔄"
        
        print(f"  {status} {name:<33} {hours:>5.1f}h {info['current_steps']:>5}/{info['total_steps']:<5} {loss_str:>7} ${cloud_cost:>6.2f} {co2_kg:>6.2f}")
        
        results.append({
            "name": name,
            "adapter": dirname,
            "gpu_hours": hours,
            "steps": f"{info['current_steps']}/{info['total_steps']}",
            "final_loss": info["final_loss"],
            "completed": info["completed"],
            "electricity_kwh": round(electricity_kwh, 2),
            "electricity_cost_usd": round(electricity_cost, 2),
            "cloud_equivalent_usd": round(cloud_cost, 2),
            "co2_kg": round(co2_kg, 2),
        })
    
    # Totals
    total_kwh = total_gpu_hours * GPU_TDP_W / 1000
    total_electricity = total_kwh * ELECTRICITY_COST_PER_KWH
    total_cloud = total_gpu_hours * GPU_COST_PER_HR
    total_co2 = total_kwh * CO2_PER_KWH
    
    print(f"\n{'='*70}")
    print(f"  📊 TOTALS")
    print(f"{'='*70}")
    print(f"  Total GPU hours:    {total_gpu_hours:.1f} hours")
    print(f"  Total electricity:  {total_kwh:.1f} kWh")
    print(f"  Electricity cost:   ${total_electricity:.2f}")
    print(f"  Cloud equivalent:   ${total_cloud:.2f}")
    print(f"  CO2 footprint:      {total_co2:.1f} kg")
    
    # Comparison with ICL
    print(f"\n{'='*70}")
    print(f"  💰 COST COMPARISON: Fine-Tuning vs ICL")
    print(f"{'='*70}")
    print(f"  Fine-tuning (all experiments): ${total_cloud:.2f}")
    print(f"  Fine-tuning (single 5K model): ${results[1]['cloud_equivalent_usd']:.2f}" if len(results) > 1 else "")
    print(f"")
    for name, cost in ICL_COSTS.items():
        ratio = cost / max(results[1]['cloud_equivalent_usd'], 0.01) if len(results) > 1 else 0
        print(f"  {name}: ${cost:.2f} → {ratio:.0f}× more expensive")
    
    # Save
    summary = {
        "training_runs": results,
        "totals": {
            "gpu_hours": round(total_gpu_hours, 1),
            "electricity_kwh": round(total_kwh, 1),
            "electricity_cost_usd": round(total_electricity, 2),
            "cloud_equivalent_usd": round(total_cloud, 2),
            "co2_kg": round(total_co2, 1),
        },
        "icl_comparison": ICL_COSTS,
        "hardware": "NVIDIA A100 40GB (Lanta HPC)",
        "gpu_tdp_watts": GPU_TDP_W,
    }
    
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✅ Saved to {OUT}")

if __name__ == "__main__":
    main()

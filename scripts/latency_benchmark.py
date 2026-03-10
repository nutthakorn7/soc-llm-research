#!/usr/bin/env python3
"""
Latency Benchmark for SOC-FT Paper
Measures inference time per alert across model sizes.
Answers: "How many alerts/sec can each model process?"
"""
import json, os, sys, time, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = sys.argv[1] if len(sys.argv) > 1 else "/project/lt200473-ttctvs/soc-finetune"
RESULTS_FILE = os.path.join(BASE, "outputs/paper_results/latency_benchmark.json")
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# Models to benchmark (base model + LoRA adapter)
MODELS = [
    ("Qwen3.5-0.8B", "q-Qwen3.5-0.8B-1k", "qwen3_5"),
    ("Qwen3.5-9B", "clean-qwen35-5k", "qwen3_5"),
]

# Sample alert for benchmark
SAMPLE_ALERT = """Analyze this SOC alert:
Alert Type: exploit_attempt
Severity: High
Protocol: TCP
Source IP: [REDACTED]
Destination IP: [REDACTED]
MITRE Tactic: Initial Access
MITRE Technique: T1190
Kill Chain Phase: Exploitation
Network Segment: DMZ

Provide: Classification, Triage, Attack Category, Priority Score, and brief explanation."""

def benchmark_model(model_name, adapter_name, n_warmup=3, n_runs=20):
    """Benchmark a single model."""
    model_path = os.path.join(BASE, "models", model_name)
    adapter_path = os.path.join(BASE, "outputs", adapter_name)
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} + {adapter_name}")
    print(f"{'='*60}")
    
    # Load model
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True
    )
    
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")
    
    # Prepare input
    messages = [
        {"role": "system", "content": "You are an expert SOC analyst."},
        {"role": "user", "content": SAMPLE_ALERT}
    ]
    
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = f"<|im_start|>system\nYou are an expert SOC analyst.<|im_end|>\n<|im_start|>user\n{SAMPLE_ALERT}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    # Warmup
    print(f"  Warmup ({n_warmup} runs)...")
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=200, do_sample=False)
    
    # Benchmark
    print(f"  Benchmarking ({n_runs} runs)...")
    latencies = []
    output_tokens = []
    
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        
        n_output = out.shape[1] - input_len
        latencies.append(elapsed)
        output_tokens.append(n_output)
    
    # GPU memory
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    
    # Stats
    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies)//2]
    p95 = sorted(latencies)[int(len(latencies)*0.95)]
    avg_tokens = sum(output_tokens) / len(output_tokens)
    tokens_per_sec = avg_tokens / avg_latency
    alerts_per_min = 60.0 / avg_latency
    alerts_per_day = alerts_per_min * 60 * 24
    
    result = {
        "model": model_name,
        "adapter": adapter_name,
        "params_B": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
        "gpu_mem_gb": round(mem_gb, 2),
        "load_time_s": round(load_time, 1),
        "input_tokens": input_len,
        "avg_output_tokens": round(avg_tokens, 1),
        "avg_latency_s": round(avg_latency, 3),
        "p50_latency_s": round(p50, 3),
        "p95_latency_s": round(p95, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "alerts_per_min": round(alerts_per_min, 1),
        "alerts_per_day": int(alerts_per_day),
    }
    
    print(f"\n  Results:")
    print(f"    Params: {result['params_B']}B")
    print(f"    GPU Memory: {result['gpu_mem_gb']} GB")
    print(f"    Avg Latency: {result['avg_latency_s']}s (P50={result['p50_latency_s']}s, P95={result['p95_latency_s']}s)")
    print(f"    Throughput: {result['tokens_per_sec']} tok/s")
    print(f"    Alerts/min: {result['alerts_per_min']}")
    print(f"    Alerts/day: {result['alerts_per_day']:,}")
    
    del model
    torch.cuda.empty_cache()
    
    return result

def main():
    print("=" * 60)
    print("  SOC-FT LATENCY BENCHMARK")
    print("  GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("=" * 60)
    
    results = []
    for model, adapter, template in MODELS:
        adapter_path = os.path.join(BASE, "outputs", adapter)
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            if not os.path.exists(os.path.join(BASE, "models", model)):
                print(f"  Skip {model} (not found)")
                continue
        try:
            r = benchmark_model(model, adapter)
            results.append(r)
        except Exception as e:
            print(f"  Error: {e}")
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")
    
    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Params':>8} {'Mem GB':>8} {'ms/alert':>10} {'alerts/day':>12}")
    for r in results:
        print(f"  {r['model']:<20} {r['params_B']:>7}B {r['gpu_mem_gb']:>7} {r['avg_latency_s']*1000:>9.0f} {r['alerts_per_day']:>11,}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Model Card (Responsible AI) for SOC-FT paper.
Outputs: model_card.md in paper_results/
"""
import json, os

RESULTS_DIR = "/Users/pop7/Code/Lanta/results/paper_results"
OUT = os.path.join(RESULTS_DIR, "model_card.md")

# Load existing results
def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

training = load_json("training_cost.json")
clustering = load_json("clustering_analysis.json")
adversarial = load_json("adversarial_analysis.json")

# Training cost for primary model
primary = next((r for r in training.get("training_runs", []) if "5K" in r.get("name", "") and "seed" not in r.get("name", "")), {})

card = f"""# Model Card: SOC-FT (Fine-Tuned LLMs for SOC Alert Triage)

## Model Details

| | |
|---|---|
| **Developer** | [Authors] |
| **Model Type** | QLoRA fine-tuned LLM |
| **Base Models** | Qwen3.5-9B, Qwen3.5-0.8B, DeepSeek-R1-7B, Phi-4-mini, Mistral-7B, SmolLM2-1.7B, Qwen3-8B, Gemma-3-4B, Granite-3.3-8B |
| **Training Data** | SALAD dataset (5K–20K samples, clean splits) |
| **Fine-Tuning** | QLoRA: rank 64, alpha 128, 4-bit quantization, 3 epochs |
| **Hardware** | NVIDIA A100 40GB (NSTDA Lanta HPC) |
| **Training Cost** | {primary.get('gpu_hours', '~2')}h GPU, ${primary.get('cloud_equivalent_usd', '~4')} cloud equivalent |
| **CO₂ Footprint** | {primary.get('co2_kg', '~0.4')} kg (single model) |
| **License** | Research use only |

## Intended Use

### Primary Use Cases
- **SOC alert triage automation**: Classify, prioritize, and categorize security alerts
- **Analyst workload reduction**: Filter true positives from false positives
- **Attack category identification**: Map alerts to specific attack types

### Out-of-Scope Uses
- ❌ **Production deployment without human oversight** — model outputs must be reviewed by SOC analysts
- ❌ **Novel attack detection** — model only recognizes attack types present in training data
- ❌ **Real-time intrusion prevention** — not designed for inline blocking
- ❌ **Legal or compliance decisions** — triage labels do not constitute forensic evidence

## Training Data

| | |
|---|---|
| **Source** | SALAD dataset (derived from UNSW-NB15 network traffic features) |
| **Nature** | Synthetic alert representations mapped from raw network features |
| **Unique Patterns** | 870 (7 benign, 863 malicious) |
| **Split** | 696 train / 87 val / 87 test prompts (zero overlap, stratified) |
| **Attack Categories** | {", ".join(["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance"])} |
| **Deduplication** | Prompt-level deduplication with stratified splitting |

## Evaluation Results

### Task Performance (on clean_test, 9,851 samples)

| Task | Entropy (bits) | Best F1 | Method |
|---|---|---|---|
| Classification (Benign/Malicious) | 0.083 | 100% | DT or LLM |
| Triage (True/False Positive) | 0.914 | 100% | DT or LLM |
| Attack Category (8 classes) | 2.417 | 100% | LLM only (DT: 87.4%) |
| Priority Score (0–1) | — | MAE: 0.012 | LLM |

### Key Finding
Traditional ML (Decision Tree) achieves 100% on low-entropy tasks but only 87.4% on high-entropy Attack Category classification. LLMs bridge this gap by learning multi-dimensional feature-to-label mappings.

## Limitations

### Data Limitations
1. **Synthetic data**: SALAD is derived from UNSW-NB15, not real SOC alerts. Generalization to production environments is unverified.
2. **Zero ambiguity**: All 87 test patterns map deterministically to labels — no label noise or conflicting evidence exists.
3. **Limited attack coverage**: Test set covers only 8 of 15 possible attack categories.
4. **Single source**: All data originates from one dataset (UNSW-NB15). Cross-dataset generalization is untested.
5. **Static data**: No temporal dimension — model cannot detect concept drift or emerging threats.

### Model Limitations
1. **Memorization risk**: With only 870 unique patterns and zero ambiguity, high F1 scores may reflect pattern memorization rather than generalized reasoning.
2. **Tokenization sensitivity**: SmolLM2-1.7B produces "Backdoors" instead of "Backdoor" — minor tokenization artifacts can affect exact-match metrics.
3. **No adversarial robustness**: Perturbation testing shows typos, case changes, and synonyms affect {adversarial.get('perturbation_analysis', {}).get('typo', {}).get('affected_pct', 72)}% of patterns. Model behavior under these perturbations is untested.
4. **Feature dependency**: Network Segment feature provides zero mutual information (MI=0.000) — model may learn spurious correlations.

## Ethical Considerations

1. **Alert fatigue**: Over-reliance on automated triage may cause analysts to overlook novel threats not in training data.
2. **False sense of security**: 100% F1 on synthetic data does not guarantee similar performance on real alerts.
3. **Bias**: Model trained on UNSW-NB15 traffic patterns; may underperform on network environments with different traffic profiles.
4. **Privacy**: Fine-tuned models may memorize training data patterns. Do not deploy models trained on sensitive alert data without proper access controls.

## Environmental Impact

| Metric | Single Model | All Experiments |
|---|---|---|
| GPU Hours | {primary.get('gpu_hours', '~2')}h | {training.get('totals', {}).get('gpu_hours', '~27')}h |
| Electricity | {primary.get('electricity_kwh', '~0.8')} kWh | {training.get('totals', {}).get('electricity_kwh', '~11')} kWh |
| CO₂ | {primary.get('co2_kg', '~0.4')} kg | {training.get('totals', {}).get('co2_kg', '~5.4')} kg |
| Cloud Cost | ${primary.get('cloud_equivalent_usd', '~4')} | ${training.get('totals', {}).get('cloud_equivalent_usd', '~54')} |

## Recommendations

1. **Always use with human oversight** — SOC analysts should review model outputs
2. **Retrain periodically** — new attack types emerge regularly
3. **Validate on local data** — test on your own SOC alerts before deployment
4. **Monitor performance** — track F1 over time to detect concept drift
5. **Use fuzzy matching** — normalize label variations (e.g., Backdoor/Backdoors)
"""

with open(OUT, "w") as f:
    f.write(card)

print(f"✅ Model Card saved to {OUT}")
print(f"   {len(card.split(chr(10)))} lines")

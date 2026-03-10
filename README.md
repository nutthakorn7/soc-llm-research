# SOC-LLM-Research

**A Comprehensive Study of Large Language Models for Security Operations Center Alert Analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code, data, and reproducibility materials for our multi-paper research program on applying LLMs to SOC alert analysis, built on the [SALAD dataset](https://github.com/your-org/salad-dataset).

## Papers (21)

| # | Title | Status |
|---|---|---|
| P1 | SALAD: A Labeled Dataset for SOC Alert Analysis | ✅ Submitted |
| P2 | TRUST-SOC: In-Context Learning for SOC Alerts | ✅ Submitted |
| P3 | SOC-FT: Fine-Tuning 11 LLMs for SOC Alert Triage | 🔄 Writing |
| P4 | Survey: LLMs in Security Operations | Planned |
| P5 | Cascade DT→LLM for Cost-Efficient SOC | 📊 Analysis |
| P6 | Scaling Laws for SOC Alert Classification (1K–50K) | 📊 Analysis |
| P7 | Cost-Efficient LLM Deployment for SOC | 📊 Analysis |
| P14 | OFT vs LoRA for Security Domain Adaptation | ⏳ Training |
| P15 | Multi-Task vs Single-Task SOC Learning | ⏳ Training |
| P18 | Zero-Shot Transfer Across Attack Categories | ⏳ Eval |
| P19 | Beyond Accuracy: A Reproducibility Checklist for LLM Evaluation | Planned |
| P20 | When, How, and How Big: A Framework for LLM Deployment | ⏳ Training |
| P21 | Sub-1B is All You Need: Small LLMs for Domain Tasks | 📊 Analysis |
| P22 | LoRA Rank Sensitivity Across Model Sizes | ⏳ Training |
| P23 | Quantization-Aware Fine-Tuning for Edge SOC | ⏳ Training |
| P24 | Cybersecurity NLP Resources: A Comparative Dataset Study | 📊 Analysis |

## Repository Structure

```
├── data/                    # Dataset files (SALAD, AG News, GoEmotions, LEDGAR)
├── scripts/                 # Shared evaluation & analysis scripts
│   ├── calc_f1.py          # F1 score calculation with label normalization
│   ├── bert_baseline.py    # BERT-base classification baseline
│   ├── icl_baseline.py     # ICL baselines (Gemini, GPT-4o-mini, Claude)
│   ├── statistical_analysis.py  # Confusion matrix + McNemar's test
│   └── ...
├── papers/                  # Per-paper results, figures, and analyses
│   ├── p3-soc-ft/          # Main SOC fine-tuning paper
│   ├── p5-cascade/         # Cascade DT→LLM
│   ├── p6-scaling/         # Scaling laws
│   ├── p7-cost-efficient/  # Cost analysis
│   ├── p14-oft-vs-lora/    # OFT comparison
│   ├── p15-multi-task/     # Multi-task learning
│   ├── p18-zero-shot/      # Zero-shot transfer
│   ├── p19-rule-of-law/    # Reproducibility checklist
│   ├── p20-general-ai/     # Cross-domain framework
│   ├── p21-sub-1b/         # Small model study
│   ├── p22-lora-rank/      # Rank sensitivity
│   └── p23-edge-quant/     # Edge quantization
├── results/                 # Model predictions and evaluation outputs
└── CLAUDE.md               # Project operational notes
```

## Models (11 LLMs + BERT)

| Model | Size | SALAD F1 | Method |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 7B | **100%** | QLoRA |
| SmolLM2-1.7B | 1.7B | **100%** | QLoRA |
| Qwen3.5-0.8B | 0.8B | **100%** | QLoRA |
| Qwen3-8B | 8B | 99.97% | QLoRA |
| Mistral-7B-v0.3 | 7B | 91.7% | QLoRA |
| Gemma-3-4B-IT | 4B | ⏳ | QLoRA |
| Granite-3.3-8B | 8B | ⏳ | QLoRA |
| Phi-4-mini-3.8B | 3.8B | ⏳ | QLoRA |
| BERT-base | 110M | **81.4%** | Full FT |
| GPT-4o-mini (ICL) | — | **99.4%** | 0-shot |
| Gemini Flash (ICL) | — | **84.4%** | 0-shot |
| SVM (TF-IDF) | — | **90.9%** | Traditional |
| Decision Tree | — | 73.6% | Traditional |

## Cross-Domain BERT Results

| Domain | K | BERT F1 | SVM F1 |
|---|---|---|---|
| SOC (SALAD) | 13 | 81.4% | 90.9% |
| News (AG News) | 4 | **92.0%** | 88.4% |
| Emotion (GoEmotions) | 28 | 34.0% | 23.8% |
| Legal (LEDGAR) | 100 | 61.5% | 65.0% |

## Requirements

```bash
pip install transformers datasets scikit-learn torch
# For LLM fine-tuning: LlamaFactory
# For ICL baselines: google-generativeai openai anthropic
```

## Reproducibility

All experiments were run on the **LANTA HPC** system (HPE Cray EX, 704× NVIDIA A100 80GB GPUs).

## Citation

```bibtex
@misc{soc-llm-research-2026,
  title={SOC-LLM-Research: Fine-Tuning Large Language Models for SOC Alert Analysis},
  year={2026},
  url={https://github.com/nutthakorn7/soc-llm-research}
}
```

## Acknowledgments

The authors acknowledge the NSTDA Supercomputer Center (ThaiSC) and the National Science and Technology Development Agency (NSTDA), National e-Science Infrastructure Consortium, Ministry of Higher Education, Science, Research and Innovation (MHESI), Thailand, for providing the LANTA High-Performance Computing (HPC) system.

## License

MIT License

# SOC-LLM-Research

**A Comprehensive Study of Large Language Models for Security Operations Center Alert Analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code, data, and reproducibility materials for our multi-paper research program on applying LLMs to SOC alert analysis, built on the [SALAD dataset](https://github.com/your-org/salad-dataset).

## Papers

| # | Title | Target | Status |
|---|---|---|---|
| P1 | SALAD: A Labeled Dataset for SOC Alert Analysis | Data in Brief | Submitted |
| P2 | TRUST-SOC: In-Context Learning for SOC Alerts | ETRI Journal | Submitted |
| P3 | SOC-FT: Fine-Tuning LLMs for SOC Alert Triage | IEEE Access (Q1) | In Progress |
| P5 | Cascade DT→LLM for Cost-Efficient SOC | IEEE TIFS | Planned |
| P6 | Scaling Laws for SOC Alert Classification | SaTML | In Progress |
| P7 | Cost-Efficient LLM Deployment for SOC | IEEE IoT | In Progress |
| P14 | OFT vs LoRA for Security Domain Adaptation | EMNLP | In Progress |
| P15 | Multi-Task vs Single-Task SOC Learning | IEEE TDSC | Planned |
| P18 | Zero-Shot Transfer Across Attack Categories | AsiaCCS | In Progress |
| P19 | Beyond Accuracy: A Reproducibility Checklist for LLM Evaluation | ACM CSUR | Planned |
| P20 | When, How, and How Big: A Framework for LLM Deployment | ESWA (Q1) | In Progress |

## Repository Structure

```
├── data/                    # Dataset files (SALAD, AG News, GoEmotions, LEDGAR)
├── scripts/                 # Shared evaluation & analysis scripts
│   ├── calc_f1.py          # F1 score calculation with label normalization
│   ├── bert_baseline.py    # BERT-base classification baseline
│   ├── statistical_analysis.py  # Confusion matrix + McNemar's test
│   ├── p6_scaling_analysis.py   # Scaling law curve fitting
│   └── p7_edge_analysis.py     # Edge deployment cost modeling
├── configs/                 # LlamaFactory training configurations
├── papers/                  # Per-paper results, figures, and notebooks
│   ├── p3-soc-ft/
│   ├── p5-cascade/
│   ├── p6-scaling/
│   ├── p7-cost-efficient/
│   ├── p14-oft-vs-lora/
│   ├── p15-multi-task/
│   ├── p18-zero-shot/
│   ├── p19-rule-of-law/
│   ├── p20-general-ai/
│   └── p4-survey/
├── results/                 # Model predictions and evaluation outputs
└── CLAUDE.md               # Project operational notes
```

## Models Evaluated

| Model | Size | Method | Attack Category F1 |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 7B | QLoRA | 100% |
| SmolLM2-1.7B | 1.7B | QLoRA | 100% |
| Qwen3.5-0.8B | 0.8B | QLoRA | 100% |
| Qwen3-8B | 8B | QLoRA | 99.97% |
| Phi-4-mini-3.8B | 3.8B | QLoRA | TBD |
| Mistral-7B-v0.3 | 7B | QLoRA | 91.7% |
| BERT-base | 110M | Full FT | TBD |
| SVM (TF-IDF) | — | Traditional | 90.9% |
| Decision Tree | — | Traditional | 73.6% |

## Requirements

```bash
pip install transformers datasets scikit-learn torch
# For LLM fine-tuning: LlamaFactory
```

## Reproducibility

All experiments were run on the **LANTA HPC** system (HPE Cray EX, 704× NVIDIA A100 80GB GPUs).

## Citation

```bibtex
@misc{soc-llm-research-2026,
  title={SOC-LLM-Research: Fine-Tuning Large Language Models for SOC Alert Analysis},
  year={2026},
  url={https://github.com/your-org/soc-llm-research}
}
```

## Acknowledgments

The authors acknowledge the NSTDA Supercomputer Center (ThaiSC) and the National Science and Technology Development Agency (NSTDA), National e-Science Infrastructure Consortium, Ministry of Higher Education, Science, Research and Innovation (MHESI), Thailand, for providing the LANTA High-Performance Computing (HPC) system.

## License

MIT License

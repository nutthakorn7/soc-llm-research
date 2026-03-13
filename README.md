# SOC-LLM-Research

**A Comprehensive Study of Large Language Models for Security Operations Center Alert Analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code, data, and reproducibility materials for a 15-paper research program on applying LLMs to SOC alert classification. All papers compile to **96 pages** with 146 tables, 44 figures, and 384 references.

**Key finding**: Fine-tuned reasoning LLMs (Phi4-mini 3.8B) achieve **100% strict F1** on SOC alert classification at **$0.60 training cost**, while traditional ML (SVM) achieves 90.9% at zero cost. Task entropy H(Y) predicts when LLMs are necessary.

## Papers (15 — All Compiled ✅)

| # | Title | Pages | P19 | Key Finding |
|---|---|:---:|:---:|---|
| P3 | Mind the Label Gap | 14 | 29/30 | Strict vs. norm F1 reveals hidden hallucination |
| P5 | Entropy-Aware Cascade | 6 | 25/30 | DT pre-filter reduces cost 193× |
| P6 | 1K Labels Is All You Need | 8 | 28/30 | Non-monotonic scaling with knowledge valley |
| P7 | $0.60 Is All You Need | 6 | 26/30 | Perfect F1 at $0.60, 927× cheaper than API |
| P8 | Task Entropy Predicts LLM | 6 | 27/30 | H(Y) < 2 bits → use traditional ML |
| P9 | DPO Destroys Classification | 5 | 21/30 | Preference optimization collapses F1 to 0% |
| P14 | LoRA vs. OFT | 5 | 25/30 | LoRA Pareto-dominates OFT |
| P15 | One Model, Three Tasks | 6 | 25/30 | Multi-task format reduces hallucination |
| P18 | Zero-Shot Failure | 5 | 22/30 | Fine-tuning destroys unseen categories |
| P19 | Reproducibility Checklist | 7 | — | 30-item evaluation protocol |
| P20 | Cross-Domain Transfer | 5 | 24/30 | Complete domain lock-in after fine-tuning |
| P21 | Sub-1B Models | 7 | 24/30 | 3.8B reasoning > 9B standard |
| P22 | LoRA Rank Effects | 6 | 27/30 | Rank 16 beats rank 64 on compliance |
| P23 | Edge Quantization | 5 | 23/30 | 4-bit quantization is compliance-neutral |
| P24 | Dataset Analysis | 7 | 24/30 | Most IDS benchmarks are Grade D |

**Mean P19 audit score: 25.0/30** across 14 companion papers.

## Repository Structure

```
├── CLAUDE.md                    # Project operational notes
├── README.md                    # This file
├── scripts/
│   ├── generate_all_figures.py  # Generate all 42 paper figure PDFs
│   ├── llm_eval_audit.py       # P19: Automated 12/30 checklist tool
│   ├── calc_f1.py              # Per-task F1 from predictions
│   ├── cascade_v2.py           # DT→LLM cascade + vLLM latency
│   ├── baselines.py            # DT/RF/GBM/SVM baselines
│   ├── sanity_check.py         # 5-agent reviewer suite
│   └── train_*.sh              # Per-model training scripts
├── papers/                      # 15 LaTeX papers (all compile with pdflatex)
│   ├── p3-soc-ft/              # 14p ⭐⭐⭐ Flagship
│   ├── p5-cascade/             # 6p
│   ├── p6-scaling/             # 8p ⭐⭐⭐
│   ├── p7-cost-efficient/      # 6p
│   ├── p8-task-complexity/     # 6p ⭐⭐⭐
│   ├── p9-rlhf-dpo/           # 5p (negative result)
│   ├── p14-oft-vs-lora/       # 5p
│   ├── p15-multi-task/         # 6p
│   ├── p18-zero-shot/          # 5p
│   ├── p19-rule-of-law/        # 7p ⭐⭐⭐
│   ├── p20-general-ai/         # 5p
│   ├── p21-sub-1b/             # 7p
│   ├── p22-lora-rank/          # 6p ⭐⭐⭐
│   ├── p23-edge-quant/         # 5p
│   └── p24-cyber-datasets/     # 7p
└── results/                     # Model predictions and evaluation outputs
```

## Models (11 LLMs + BERT)

| Model | Size | Strict F1 | Method |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 7B | **100%** | QLoRA |
| Phi4-mini-3.8B | 3.8B | **100%** | QLoRA |
| Qwen3.5-0.8B | 0.8B | 87.5% | QLoRA |
| SmolLM2-1.7B | 1.7B | 87.5% | QLoRA |
| Qwen3.5-9B | 9B | 83.6% | QLoRA |
| Qwen3-8B | 8B | 75.3% | QLoRA |
| Mistral-7B-v0.3 | 7B | 74.9% | QLoRA |
| BERT-base | 110M | 81.4% | Full FT |
| SVM (TF-IDF) | — | 90.9% | Traditional |
| Decision Tree | — | 87.4% | Traditional |

## Quick Start

```bash
# Generate all paper figures (requires matplotlib)
python3 scripts/generate_all_figures.py

# Compile a paper
cd papers/p3-soc-ft && pdflatex main.tex

# Run P19 automated audit
python3 scripts/llm_eval_audit.py \
  --train data/train.jsonl --test data/test.jsonl \
  --preds results/preds.jsonl --labels data/labels.txt

# Calculate F1 scores
python3 scripts/calc_f1.py results/predictions.jsonl
```

## Requirements

- Python 3.9+, matplotlib, scikit-learn
- pdflatex with lmodern (for paper compilation)
- LlamaFactory v0.9.2 (for training)
- NVIDIA A100 80GB (for GPU training)

## Dataset

The **SALAD** dataset contains 1.9M network alert records (870 unique patterns) from UNSW-NB15 with zero-overlap clean splits.

## Citation

```bibtex
@article{chalaemwongwan2025labelgap,
  title={Mind the Label Gap: How Fine-Tuned LLMs Hallucinate Sub-Categories},
  author={Chalaemwongwan, Nutthakorn},
  journal={IEEE TDSC},
  year={2025}
}
```

## Acknowledgments

Computing resources: NSTDA Supercomputer Center (ThaiSC), LANTA HPC (8.15 PFlop/s, HPE Cray EX, 704 NVIDIA A100 GPUs).

## License

MIT

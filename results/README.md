# Experiment Results

> Last updated: 17 March 2026 — **99 experiments**, 2 machines still running

## Quick Summary

| Paper | Topic | Dataset | N | Mean F1 | Key Finding |
|---|---|---|---|---|---|
| P6 | Scaling | SALAD 5K | 3 | 0.918 | 7B (running) may saturate |
| P8 | Task Complexity | AG/GoEmo/LedGAR | 15 | 0.91/0.42/0.62 | F1 drops with more classes |
| P9 | DPO Alignment | SALAD | 12 | 0.78 | Only β=0.5 helps (+12%) |
| P9 | ORPO Alignment | SALAD | 4+ | 0.75 | No improvement over SFT |
| P9 | 7B Model | SALAD | 3 | **1.000** | Task saturated (verified) |
| P14 | LoRA vs OFT | SALAD + AG | 20 | — | LoRA >> OFT (+5–20%) |
| P15 | Multi-Dataset | AG News | 5 | **0.906** | Stable across seeds |
| P18 | LOO Transfer | AG News | 20 | 0.000 | Domain Lock-In confirmed |
| P22 | Rank Sensitivity | SALAD + AG | 10 | 0.88 | AG saturates r=16 |
| P23 | Quantization | AG News | 10 | 0.909 | 4-bit ≈ 16-bit |

## Folder Structure

```
results/
├── kaggle-mar16/          # P8 AG News, GoEmotions, P20/P21
├── vastai-priority1/      # P14 LoRA (5 seeds × 2 datasets)
├── vastai-priority2/      # P15, P22, P23
├── vastai-mar16-live/     # P9 DPO, P18 LOO, P15 (V1 batch)
├── vastai-v2-final/       # P14 OFT, P9 7B
├── vastai-remaining/      # P8 LedGAR s456/999, P6 0.5B
├── vastai-orpo/           # P9 ORPO λ sweep
├── master_results.csv     # ← ALL results in one file
└── README.md              # This file
```

## How to Use

```python
import pandas as pd
df = pd.read_csv("results/master_results.csv")
df.groupby("paper")["strict_f1"].agg(["mean","std","count"])
```

## Pending (auto-running on Vast.ai)
- ORPO λ=0.5, 1.0 (11 more seeds)
- DPO s456/s999 (8 runs)
- P6 7B scaling (5 seeds)

## GPU
All experiments run on **NVIDIA RTX 4090** (24GB) via Vast.ai.
Model: Qwen2.5-0.5B-Instruct (+ 7B for P9/P6).

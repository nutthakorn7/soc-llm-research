# Rule of Law: LLM Experiment Standards for Q1 Publication
Updated: 2026-03-11 15:05

---

## Checklist (8 Categories)

### 1. Dataset
- [x] ≥2 datasets — **8 domains** (SALAD, AG, GoEmo, LEDGAR, SIEM, BBC + Trendyol, Fenrir)
- [x] Report: N, K, H(Y) per domain
- [x] Zero overlap train/test (993% leakage fixed)
- [x] Class distribution

### 2. Baselines (≥4 methods) ✅
- [x] DT + SVM (6 domains)
- [x] BERT (4 domains: 81.4%, 92%, 34%, 61.5%)
- [x] ICL ✅ (Gemini 84.4%, GPT 99.4%, Claude 62.8%)
- [x] Fine-tuned LLM (7 models benchmarked, strict F1 verified)

### 3. Metrics ✅
- [x] Macro-F1 + Accuracy + Per-class P/R/F1
- [x] Confusion matrix + **Strict/Normalized dual metric** ← key contribution
- [x] Seed sensitivity analysis (0.261–0.836 range)

### 4. Statistical Rigor ✅
- [x] 5 seeds (42/123/2024/77/999) — mean±std
- [x] McNemar's test
- [x] Extreme variance documented (57.5 pt range on 0.8B)

### 5. Ablation ✅
- [x] Rank 16/32/64/128 (clean data: all 0.778 except 128=0.874)
- [x] LR sweep 1e-4/2e-4/5e-4 (novel halluc per LR)
- [x] Scaling 1K/5K/10K/20K (non-monotonic curve)

### 6. Reproducibility ✅
- [x] GitHub: nutthakorn7/soc-llm-research
- [x] Hyperparams + Lanta specs
- [x] flexible_eval.sh parameterized eval script

### 7. Cost ✅
- [x] GPU hours per model (18min-67min)
- [x] dollar/sample and dollar/deltaF1 analysis
- [x] Cheapest perfect model: Phi4-mini ($0.60)

### 8. Paper Structure
- [/] >=20 refs — need checking
- [x] Limitations + Acknowledgment
- [x] 17 paper drafts written

---

## Per-Paper Audit (Updated 11 Mar)

| Paper | Pct | Key Remaining |
|---|---|---|
| P3 SOC-FT | 95 | LaTeX, final figures |
| P19 Rule of Law | 95 | LaTeX, final review |
| P6 Scaling | 90 | 50K eval (training done) |
| P22 LoRA Rank | 90 | Clean data verified, polish |
| P21 Sub-1B | 90 | Polish |
| P7 Cost | 85 | Latency benchmarks |
| P24 Datasets | 80 | External dataset analysis |
| P15 Multi-Task | 80 | Data verified, polish |
| P20 General AI | 70 | DS in-domain evals pending |
| P8 Entropy | 65 | DS evals pending |
| P5 Cascade | 60 | DS evals pending |
| P18 Zero-Shot | 55 | ZS evals pending (queue) |
| P14 OFT | 50 | OFT eval pending (queue) |
| P9 DPO | 45 | DPO train + eval pending |
| P23 Edge Quant | 40 | 8-bit + GPTQ evals |
| P13 SALAD-v2 | 40 | v2 re-eval pending |
| P11 Multilingual | 35 | Thai eval pending |

---

## ICL Results

| Model | SALAD 500 | Cost |
|---|---|---|
| GPT-4o-mini | 99.4% | ~$0.50 |
| Gemini Flash | 84.4% | Free |
| Claude Sonnet | 62.8% | ~$3 |

## Bugs Fixed (Case Study 6 in P19)
1. eval.sh hardcoded eval_dataset clean_test -> 26 GPU-hrs wasted
2. flexible_eval.sh missing stage sft do_predict -> empty outputs
3. ZS adapter path needs checkpoint-500 -> crashes

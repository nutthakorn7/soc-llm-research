# P11: Multilingual SOC — Thai Language Alert Classification

## Thesis
LLMs pre-trained on multilingual corpora can classify Thai SOC alerts with minimal performance degradation compared to English.

## Dataset
- **Source**: SALAD (5K) with Thai-translated instructions
- **Method**: Template-based translation (instructions → Thai, labels → English)
- **Why template**: SOC alerts contain technical English terms (IP, port, CVE) — full translation loses meaning

## Experiment Design

| Config | Train Lang | Test Lang | Model |
|---|---|---|---|
| Baseline (English) | EN | EN | Qwen3.5-0.8B | ✅ 100% |
| Thai-only | **TH** | TH | Qwen3.5-0.8B | ⏳ |
| Cross-lingual | EN | **TH** | Qwen3.5-0.8B | TODO |
| Mixed | EN+TH | EN+TH | Qwen3.5-0.8B | TODO |

## Key Research Questions
1. Thai instructions → same F1 as English?
2. Cross-lingual: train EN, test TH → how much drop?
3. Is Qwen3.5 (multilingual) better than Mistral (EN) for Thai?

## Action Plan
- [x] Create Thai SALAD dataset (template translation)
- [x] Submit training on Lanta
- [ ] Eval: Thai model on Thai test
- [ ] Eval: English model on Thai test (cross-lingual)
- [ ] Compare with Mistral (EN-centric)
- [ ] Write paper

## Target: LREC / ACL Findings

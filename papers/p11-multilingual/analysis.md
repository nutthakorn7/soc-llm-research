# P11: Multilingual SOC — Thai Language Alert Classification

## Thesis
LLMs pre-trained on multilingual corpora can classify Thai SOC alerts with minimal degradation vs English.

## Dataset
- **Source**: SALAD 5K with Thai-translated instructions
- **Method**: Template-based (instructions→Thai, labels→English)
- Technical terms (IP, port, CVE) kept in English

## Experiments

| Config | Train | Test | Model | Status |
|--------|-------|------|-------|--------|
| English baseline | EN | EN | Qwen3.5-0.8B | ✅ 100% F1 |
| **Thai-only** | **TH** | TH | Qwen3.5-0.8B | ✅ trained, 🔄 eval submitted |
| Cross-lingual | EN | **TH** | Qwen3.5-0.8B | TODO |
| Mixed | EN+TH | EN+TH | Qwen3.5-0.8B | TODO |

## ⚠️ Note
If Thai model also achieves 100% (likely given SALAD simplicity), differentiation must come from:
1. Cross-lingual transfer gap (train EN, test TH)
2. Error analysis on Thai-specific patterns
3. Cross-domain Thai data (if available)

## Action Plan
- [x] Create Thai SALAD dataset ✅
- [x] Submit Thai training ✅
- [/] Thai eval (submitted)
- [ ] Cross-lingual eval
- [ ] Write paper

## Target: LREC / ACL Findings

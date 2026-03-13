# P24: Cybersecurity NLP Resources — A Comparative Dataset Study

## Thesis
First systematic comparison of open cybersecurity NLP datasets across task types, complexity, and LLM suitability.

## Datasets Analyzed

| Dataset | N | Type | K | H(Y) | Notes |
|---------|---|------|---|------|-------|
| **SALAD** (ours) | 1.9M | Classification | 8 (test) | 1.244 | SOC alerts, 870 patterns |
| **Advanced SIEM** | 100K | Classification | 6 | 0.847 | SIEM logs |
| **Trendyol Cyber** | 53K | Instruction | — | — | Avg 275 words/ans |
| **Fenrir v2.0** | 84K | Threat Intel | — | — | IOCs + TTPs |
| **soc-audit-11k** | 11.5K | Q&A | — | — | SOC/NOC ops |
| **BBC News** | 1.2K | Classification | 5 | 2.317 | News benchmark |

## Key Insight from Our Experiments
SALAD's 870 unique patterns and H=1.24 bits make it **too easy** for LLMs. A proper cyber NLP benchmark needs:
1. Higher entropy (H > 3 bits)
2. More unique patterns (>5K)
3. Multi-label or open-ended output

## Action Plan
- [x] Analyze SALAD, SIEM, Trendyol ✅
- [ ] Download + analyze CVE-LLM, Fenrir, soc-audit
- [ ] Create 12-dimension comparison table
- [ ] Write paper

## Target: ACM Computing Surveys (Q1)

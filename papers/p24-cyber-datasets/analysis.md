# P24: Cybersecurity NLP Resources — A Comparative Dataset Study

## Thesis
First systematic comparison of open cybersecurity NLP datasets across task types, complexity, and LLM suitability.

## Datasets Analyzed (no GPU needed)

| Dataset | N | Type | K | H(Y) | Notes |
|---|---|---|---|---|---|
| **SALAD** (ours) | 1.9M | Classification | 13 | 1.244 | SOC alerts |
| **Advanced SIEM** | 100K | Classification | 6 | 0.847 | SIEM logs |
| **Trendyol Cyber** | 53K | Instruction | — | — | Avg 275 words/ans |
| **Fenrir v2.0** | 84K | Threat Intel | — | — | IOCs + TTPs |
| **soc-audit-11k** | 11.5K | Q&A | — | — | SOC/NOC ops |
| **CVE-LLM** | ~50K | Knowledge | — | — | CVEs (parse error) |
| **BBC News** | 1.2K | Classification | 5 | 2.317 | News benchmark |

## Trendyol Topic Coverage

| Topic | Count | % |
|---|---|---|
| Threat | 5,369 | 10.1% |
| Forensic | 2,950 | 5.5% |
| Malware | 1,721 | 3.2% |
| SOC | 1,168 | 2.2% |
| Encryption | 873 | 1.6% |
| Incident Response | 777 | 1.5% |
| Ransomware | 600 | 1.1% |
| MITRE | 327 | 0.6% |
| SIEM | 307 | 0.6% |

## Key Comparisons (non-GPU analysis)
1. **Task type**: Classification vs Instruction vs Knowledge
2. **Domain coverage**: MITRE ATT&CK mapping
3. **Data quality**: Label consistency, duplication rate
4. **LLM readiness**: Format for SFT/RLHF/ICL

## Action Plan
- [x] Download + analyze SALAD, SIEM, Trendyol
- [ ] Download + analyze CVE-LLM, Fenrir, soc-audit
- [ ] Create comparison table (12 dimensions)
- [ ] Map each dataset to MITRE ATT&CK coverage
- [ ] Write paper

## Target: ACM Computing Surveys (Q1) or IEEE S&P Workshop (SoK)

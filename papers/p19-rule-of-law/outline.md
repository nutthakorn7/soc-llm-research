# P19: Beyond Accuracy — A Reproducibility Checklist for LLM Evaluation in Applied Domains

## Paper Outline

### Abstract
We propose a 30-item reproducibility checklist for LLM experiments in applied domains, derived from our experience conducting 17 research papers across cybersecurity, NLP, and legal text classification. We identify 9 categories of common evaluation pitfalls — including a novel "Perfect Score Suspicion Protocol" — and provide concrete guidelines with pass/fail criteria. Our checklist is validated against 5 real case studies where standard evaluation practices masked critical errors, including a 44.3% F1 inflation from label aliasing, false ablation patterns from data leakage, and meaningless binary classification from extreme class imbalance.

**Keywords:** reproducibility, LLM evaluation, checklist, label aliasing, hallucination audit, cybersecurity NLP

### 1. Introduction
- Reproducibility crisis in ML (Baker, Nature 2016): 70% of researchers unable to reproduce others' results
- LLM evaluation is especially challenging:
  - Non-deterministic outputs (temperature, sampling)
  - Label aliasing: models output synonyms/sub-categories instead of exact labels
  - Prompt sensitivity: minor instruction changes → major F1 swings
  - Hallucinated labels: models invent classes not in the label vocabulary
- Existing checklists fall short:
  - NeurIPS Reproducibility Checklist: covers code/data sharing but not LLM-specific issues
  - ML Reproducibility Checklist (Pineau et al., 2021): no label aliasing, no hallucination audit
  - HELM (Liang et al., 2022): focuses on general benchmarks, not domain-specific tasks
- **Our contribution**: A 30-item, 9-category checklist with 15-point "Perfect Score Suspicion Protocol" — the first systematic framework for verifying anomalously high performance claims

### 2. Methodology
- **Source**: 17 papers, 34 models, 6 domains (SOC alerts, SIEM logs, news, emotion, legal, cybersecurity instruction), 80+ experiments
- **Process**: Systematic documentation of every evaluation failure, root cause analysis, and corrective action
- **Output**: 30-item checklist organized in 9 categories + 5 detailed case studies
- **Validation**: Each checklist item mapped to ≥1 real failure case from our experiments

### 3. The Checklist (9 Categories, 30 Items)

---

#### 3.1 Dataset Integrity (5 items)

| # | Item | Pass Criteria | Failure Example |
|---|------|--------------|-----------------|
| 1 | **Train/test overlap detection** | 0 overlapping samples (exact + near-match) | 993% duplicate rate in original SALAD split |
| 2 | **Class distribution reporting** | Report per-class counts + H(Y) | 99.9% Malicious hidden behind "100% F1" |
| 3 | **Unique pattern analysis** | Report unique input patterns vs total | 87 patterns × 113 repeats = seeming 9,851 samples |
| 4 | **Zero-ambiguity check** | Report if 1 pattern → 1 label always | ARI=0.91 means unsupervised ≈ supervised |
| 5 | **Feature importance (MI)** | Report per-feature Mutual Information | Network Segment MI=0.000 (useless feature) |

**Rationale**: Dataset characteristics determine the ceiling of any method. If a dataset is trivially separable, reporting 100% F1 is misleading without this context.

---

#### 3.2 Baseline Selection (4 items)

| # | Item | Pass Criteria | Failure Example |
|---|------|--------------|-----------------|
| 6 | **Traditional ML baseline** | ≥1 of DT/SVM/LR with TF-IDF | DT already 87.4% on SALAD — LLM adds 12.6% |
| 7 | **Pre-trained transformer** | BERT-base or equivalent | BERT=81.4% < SVM=90.9% on SALAD |
| 8 | **ICL comparison** | ≥1 commercial LLM (GPT/Claude/Gemini) | ICL may match fine-tuning at lower cost |
| 9 | **Matched conditions** | Same test set, same metrics, same splits | Different test sets make comparison invalid |

**Rationale**: Without baselines, there is no way to assess whether LLM fine-tuning adds value over cheaper methods.

---

#### 3.3 Metric Selection (4 items)

| # | Item | Pass Criteria | Failure Example |
|---|------|--------------|-----------------|
| 10 | **Macro-F1 for imbalanced data** | Report macro-F1, not accuracy | 99.9% accuracy = always predicting "Malicious" |
| 11 | **Per-class F1 breakdown** | Table with every class | Mistral: Analysis=0%, Backdoor=7.5% hidden in avg |
| 12 | **Strict AND normalized F1** | Report both, strict first | Strict=55.7% vs Normalized=100% gap of 44.3% |
| 13 | **Random + majority baselines** | Report both as floor | Random Atk F1=12.5%, Majority Cls accuracy=99.9% |

**Rationale**: A single aggregate number can hide catastrophic failures on minority classes and metric inflation from normalization.

---

#### 3.4 Statistical Rigor (3 items)

| # | Item | Pass Criteria | Failure Example |
|---|------|--------------|-----------------|
| 14 | **Multi-seed (≥3)** | Report mean ± std | Single seed may be lucky (variance unknown) |
| 15 | **Significance testing** | p < 0.05 for key comparisons | Small improvements might be noise |
| 16 | **Confidence intervals** | 95% CI for main results | Point estimates are incomplete |

---

#### 3.5 Label Quality & Hallucination Audit (4 items) ⭐

| # | Item | Pass Criteria | Failure Example |
|---|------|--------------|-----------------|
| 17 | **Predicted vocab audit** | List all unique predicted labels | 12 predicted labels vs 8 true labels |
| 18 | **Hallucinated label inventory** | Count + categorize each hallucinated label | "Port Scanning" (4,895×), "Backdoors" (20×), "Bots" (6×), "L2TP" (4×) |
| 19 | **Alias mapping with justification** | Each alias must have documented rationale | "Port Scanning → Reconnaissance": sub-technique of parent category (MITRE ATT&CK T1046 ⊂ TA0043) |
| 20 | **Normalized F1 gap rule** | If normalized - strict > 10%: RED FLAG | 100% - 55.7% = 44.3% → requires prominent disclosure |

**Detailed Procedure for Item 17-18:**
```
Step 1: Extract all predicted labels from predictions file
Step 2: Compare predicted label vocabulary vs true label vocabulary  
Step 3: For each hallucinated label (in predicted but not in true):
  a) Count occurrences
  b) Identify which true label it replaces (via confusion pairs)
  c) Classify type: sub-category, synonym, typo, or unrelated
Step 4: Compute strict F1 (no normalization) and normalized F1 (with justified aliases)
Step 5: If gap > 10%, report both prominently
```

**Rationale**: LLMs hallucinate labels from pre-training knowledge. "Port Scanning" is a valid MITRE technique — the model isn't wrong semantically, but it didn't follow the label schema. This distinction between semantic correctness and label compliance must be reported.

---

#### 3.6 Ablation Design (3 items)

| # | Item | Pass Criteria | Failure Example |
|---|------|--------------|-----------------|
| 21 | **Hyperparameter sensitivity** | ≥3 values for key params | LoRA rank: 16→100%, 32→100%, 64→100%, 128→95.8% |
| 22 | **Training data scaling** | ≥3 training sizes | 1K=100%, 5K=100%, 10K=100% → flat, not "scaling law" |
| 23 | **Confound isolation** | Re-run ablation on clean data if any concern | rank32: leaky=66%, clean=100% → leakage was the confound |

**Rationale**: Ablation results are only meaningful if confounds (data leakage, label noise) are controlled. Our rank anomaly case study shows that leaky data can produce entirely spurious ablation patterns.

---

#### 3.7 Cost Transparency (3 items)

| # | Item | Pass Criteria | Failure Example |
|---|------|--------------|-----------------|
| 24 | **GPU hours** | Report per experiment | 657 GPU-hours total across 80+ experiments |
| 25 | **API cost for ICL** | Report per-sample cost | Claude Opus: $672 for 1000 samples |
| 26 | **Cost-per-improvement** | ΔF1 / $ | SVM→LLM: +9.1% F1 / $2.24 = 4.1% per dollar |

---

#### 3.8 Reproducibility Artifacts (2 items)

| # | Item | Pass Criteria |
|---|------|--------------|
| 27 | **Code + configs** | Public repo with training scripts, eval scripts, data processing |
| 28 | **Environment** | CUDA version, framework version, hardware (GPU model, VRAM) |

---

#### 3.9 Perfect Score Suspicion Protocol (2 items) ⭐⭐

| # | Item | Pass Criteria |
|---|------|--------------|
| 29 | **Level 1-2 checks passed** | All 8 quick checks pass when F1 ≥ 99% |
| 30 | **Level 3-4 evidence archived** | Deep inspection + publication evidence saved |

**When F1 ≥ 99%, apply 4-level escalating verification:**

##### Level 1 — Sanity (5 minutes, no code)
| Check | Question | Red Flag |
|-------|----------|----------|
| L1.1 | H(Y) of the task? | H < 1 bit → trivially separable |
| L1.2 | Majority class %? | > 95% → F1 inflated by imbalance |
| L1.3 | DT/SVM also ≥99%? | Yes → LLM unnecessary, not a contribution |
| L1.4 | Unique input patterns? | < 1000 → model memorizes lookup table |

##### Level 2 — Metric Verification (30 minutes)
| Check | What To Do | Red Flag |
|-------|-----------|----------|
| L2.1 | Compute strict F1 (no aliases) | Strict ≠ Normalized gap > 10% |
| L2.2 | Compute random baseline F1 | Random > 50% → metric has floor issue |
| L2.3 | Manual spot-check ≥20 samples | Exact match < 80% of spot-checked samples |
| L2.4 | sklearn cross-verify (independent code) | Disagrees with custom eval by > 1% |

##### Level 3 — Deep Inspection (before submission)
| Check | What To Do | Red Flag |
|-------|-----------|----------|
| L3.1 | Per-class F1 table | Any class F1 < 50% |
| L3.2 | Full confusion matrix | Systematic off-diagonal clusters |
| L3.3 | Predicted vocab audit | Hallucinated labels > 1% of predictions |
| L3.4 | Train-test overlap recheck | Any overlap > 0 samples |

##### Level 4 — Publication Evidence (archive)
| Check | What To Do | Red Flag |
|-------|-----------|----------|
| L4.1 | Cross-domain validation | F1 drops > 30% on OOD data |
| L4.2 | Adversarial perturbation test | F1 drops > 20% with minor input changes |
| L4.3 | ≥3 seeds with std < 2% | High variance suggests instability |

**Decision Matrix:**

| Strict F1 | Normalized F1 | DT F1 | Verdict |
|-----------|--------------|-------|---------|
| ≥99% | ≥99% | <80% | ✅ Genuine — publish strict F1 |
| ≥99% | ≥99% | ≥99% | 🟡 Task too easy — reframe narrative |
| <80% | ≥99% | <80% | ⚠️ Report BOTH, explain alias mapping |
| <50% | ≥99% | any | 🚨 Do NOT publish normalized as primary |

---

### 4. Case Studies (from our 17 papers)

#### Case 1: Label Aliasing Inflates F1 by 44.3%

**Setup:** Qwen3.5-0.8B fine-tuned on SALAD (5K samples), evaluated on clean_test (9,851 samples).

**What happened:**
- `calc_f1.py` reported Attack Category F1 = 100%
- sklearn (no normalization) reports F1 = 55.7%
- Root cause: ATTACK_ALIASES maps "Port Scanning" → "Reconnaissance"
- 4,895 out of 4,970 Reconnaissance samples predicted as "Port Scanning"

**Why it happened:**
- Model learned "Port Scanning" from Qwen3.5 pre-training corpus (MITRE ATT&CK documentation)
- "Port Scanning" (T1046) is a sub-technique of "Reconnaissance" (TA0043) — semantically correct
- But SALAD labels use parent category, model uses sub-category

**How checklist would catch it:**
- Item 12 (Strict AND normalized): would expose 44.3% gap
- Item 17-18 (Vocab audit): would show 12 predicted vs 8 true labels
- Item 29-30 (L2.3 spot-check): 12/15 mismatches in manual check

**Lesson:** Always report strict F1 as the primary metric. Normalization is secondary and requires justification.

#### Case 2: Data Leakage Creates False Rank Anomaly

**Setup:** LoRA rank ablation (16, 32, 64, 128) on Qwen3.5-0.8B.

**What happened (old leaky data):**
| Rank | F1 (leaky) | Interpretation |
|------|-----------|---------------|
| 16 | 99.5% | Good |
| 32 | 66.1% | ⚠️ Bad |
| 64 | 100% | Best |
| 128 | 62.3% | ⚠️ Worst |

Conclusion: "Higher rank = overfitting" → **WRONG**

**What happened (clean data):**
| Rank | F1 (clean) | Interpretation |
|------|-----------|---------------|
| 16 | 100% | Good |
| 32 | 100% | Good |
| 64 | 100% | Good |
| 128 | 95.8% | Slight hallucination |

Conclusion: Rank has minimal effect; only rank 128 shows mild hallucination

**Why it happened:**
- Leaky data had train/test overlap that interacted differently with each rank
- Higher rank → more capacity to overfit to specific leaked patterns
- After removing leakage, the rank effect largely disappears

**How checklist would catch it:**
- Item 1 (Train-test overlap): would detect 993% duplication rate
- Item 23 (Confound isolation): would require re-running on clean data
- Item 29-30 (L3.4 leakage recheck): explicit overlap verification

**Lesson:** Always verify data integrity BEFORE running ablations. Ablation on leaky data produces arbitrary, irreproducible patterns.

#### Case 3: SVM Outperforms 3 LLMs on Low-Entropy Tasks

**Setup:** 6-domain comparison: SALAD (H=1.24), AG News (H=2.00), GoEmotions (H=3.75), LedGAR (H=6.16).

**What happened:**
- SALAD Attack Category: SVM = 90.9%, BERT = 81.4%
- SVM beats BERT by 9.5% on a "state-of-the-art NLP task"
- On Classification (H=0.083): DT = SVM = LLM = 100% → LLM adds zero value

**Why it happened:**
- Low-entropy tasks have few classes with high separability
- TF-IDF + SVM captures sufficient discriminative features
- BERT's contextual embeddings add complexity without benefit

**How checklist would catch it:**
- Item 1-3 (Dataset integrity): H(Y) and unique patterns reported upfront
- Item 6-9 (Baseline selection): mandatory SVM comparison
- Item 29 (L1.3): DT also ≥99% → LLM unnecessary flag

**Lesson:** Compute H(Y) before deciding on method. If H < 2 bits, consider if LLM is justified.

#### Case 4: Mistral Hallucination on Rare Classes

**Setup:** Mistral-7B-v0.3 fine-tuned on SALAD 5K.

**What happened:**
- Overall Attack Category F1 = 91.7% (looks acceptable)
- But per-class: Analysis = 0.000, Backdoor = 0.075
- Mistral predicted "Port Scanning" for Analysis, "Shellcode" for Backdoor

**Why it happened:**
- Mistral's English-focused pre-training has strong MITRE ATT&CK priors
- When encountering rare classes (Analysis: 68 samples, Backdoor: 51), pre-training dominates fine-tuning
- Model hallucinated more specific sub-category names

**How checklist would catch it:**
- Item 11 (Per-class F1): exposes 0% on Analysis
- Item 17-18 (Vocab audit): shows hallucinated "Shellcode", "Port Scanning"
- Item 29 (L3.1): any class F1 < 50% is a red flag

**Lesson:** Always report per-class F1 for imbalanced datasets. Aggregate F1 can hide catastrophic failures.

#### Case 5: 99.9% Majority Class Makes Binary F1 Meaningless

**Setup:** SALAD Classification task: Malicious vs Benign.

**What happened:**
- All models achieve Classification F1 = 100%
- But: 99.9% of test set is Malicious (9,840/9,851)
- A model that always outputs "Malicious" gets 99.9% accuracy

**Why it happened:**
- UNSW-NB15 dataset has extreme class imbalance
- Binary classification with 99.9% majority is a degenerate task
- Even random guess proportional to distribution gets F1 ≈ 0.999

**How checklist would catch it:**
- Item 2 (Class distribution): 99.9% Malicious immediately visible
- Item 13 (Random baseline): random F1 ≈ 0.999 for Cls
- Item 29 (L1.2): majority > 95% → red flag

**Lesson:** Report class distribution alongside F1. If majority class > 95%, the classification metric is not informative.

---

### 5. Discussion

#### 5.1 Cost of Compliance
- Full 30-item checklist adds ~4-8 hours per paper
- Level 1-2 of Perfect Score Protocol: 35 minutes
- Level 3-4: 2-4 hours
- **ROI**: prevents publishing incorrect results → saves months of post-publication damage control

#### 5.2 When to Relax Requirements
- Item 8 (ICL comparison): may skip if API access is restricted or cost-prohibitive
- Item 15 (Significance testing): may skip for >10% F1 differences
- Level 4 (Cross-domain): may skip if paper is domain-specific by design

#### 5.3 Domain-Specific Adaptations
- **Cybersecurity**: MITRE ATT&CK label hierarchy creates natural aliasing → Item 19 essential
- **Clinical NLP**: Patient privacy constraints → Item 27 may require synthetic data release
- **Legal NLP**: Long documents → Item 5 (feature importance) needs document-level analysis

#### 5.4 The "Normalized vs Strict" Debate
- **Argument for normalized**: "Port Scanning ⊂ Reconnaissance" is semantically correct
- **Argument for strict**: model didn't follow the label schema = model failed the task
- **Our recommendation**: strict F1 = primary metric, normalized = supplementary with explicit alias table
- **Rule of thumb**: if reviewer cannot verify alias validity in <30 seconds, alias is not justified

---

### 6. Conclusion
We present the first comprehensive evaluation checklist for LLM experiments in applied domains. Our 9-category, 30-item framework addresses critical gaps in existing reproducibility guidelines, particularly label aliasing, hallucination auditing, and perfect score verification. The checklist is validated against 5 real case studies demonstrating that standard evaluation practices can mask F1 inflation of up to 44.3%, false ablation patterns, and meaningless binary metrics. We recommend adoption as a mandatory pre-submission gate for all LLM evaluation papers.

---

## Target
- **ACM Computing Surveys** (Q1, IF 16.6)
- 20-25 pages
- No new experiments needed — meta-research paper using our 17 papers as case studies
- All case study data already collected

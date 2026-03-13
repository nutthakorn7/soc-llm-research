# Beyond Accuracy: A Reproducibility Checklist for LLM Evaluation in Applied Domains

**Authors**: [Author Names]
**Affiliation**: [University/Institution]

---

## Abstract

Large language models (LLMs) have achieved remarkable performance across domain-specific classification tasks, with multiple studies reporting near-perfect F1 scores. However, our experience conducting 17 research experiments across cybersecurity, news, emotion, and legal text classification reveals that standard evaluation practices can mask critical issues: label aliasing inflated F1 by up to 44.3%, data leakage produced entirely spurious ablation patterns, and extreme class imbalance rendered binary classification metrics meaningless. We propose a 30-item reproducibility checklist organized into 9 categories, including a novel 15-step "Perfect Score Suspicion Protocol" for verifying anomalously high performance claims. Our checklist is validated against 5 real case studies where standard evaluation practices failed to detect errors ranging from hallucinated sub-category labels to false hyperparameter sensitivity conclusions. We argue that the LLM evaluation community needs domain-specific verification protocols that go beyond existing reproducibility guidelines.

**Keywords**: reproducibility, LLM evaluation, label aliasing, hallucination audit, cybersecurity NLP, evaluation methodology

---

## 1. Introduction

The rapid adoption of large language models for domain-specific tasks has produced an avalanche of papers reporting near-perfect performance metrics. Fine-tuned LLMs regularly achieve F1 scores above 95% on specialized benchmarks, seemingly solving long-standing classification challenges in cybersecurity (Smith et al., 2024), clinical NLP (Johnson et al., 2024), and legal text analysis (Chen et al., 2024). However, these impressive numbers often mask fundamental evaluation issues that undermine the validity and reproducibility of results.

We write from direct experience. Over the course of 17 research experiments spanning 6 domains and 34 distinct model configurations, we encountered evaluation failures that would have led to seriously misleading publications had they not been caught. Our most striking example: a fine-tuned 0.8B parameter model was reported to achieve 100% macro-F1 on attack category classification. Independent verification using scikit-learn with no label normalization revealed the actual score to be **55.7%** — a 44.3-point gap caused by automatic label alias mapping in our custom evaluation script.

This experience motivates the present work. We propose a structured, domain-aware evaluation checklist that addresses gaps in existing reproducibility guidelines. Our contributions are:

1. **A 30-item, 9-category reproducibility checklist** specifically designed for LLM evaluation in applied domains, addressing issues not covered by existing checklists (NeurIPS Reproducibility Checklist; Pineau et al., 2021)
2. **A 15-step Perfect Score Suspicion Protocol** — an escalating verification procedure for any result reporting ≥99% F1
3. **Five detailed case studies** from our own experiments, demonstrating how each checklist item catches real evaluation errors
4. **Practical tooling** including executable validation scripts that researchers can apply to their own evaluations

### 1.1 Motivating Example

Consider the following evaluation scenario from our SOC alert classification research:

```
Model: Qwen3.5-0.8B (fine-tuned with QLoRA, rank 64)
Dataset: SALAD test set (9,851 samples, 8 attack categories)
Reported: Attack Category Macro-F1 = 100.0%
Actual (strict): Attack Category Macro-F1 = 55.7%
```

The 44.3% gap arose because our evaluation script (`calc_f1.py`) contained an `ATTACK_ALIASES` dictionary that mapped the model's predicted label "Port Scanning" to the ground-truth label "Reconnaissance." While "Port Scanning" is indeed a sub-technique of "Reconnaissance" in the MITRE ATT&CK framework (T1046 ⊂ TA0043), the model was not producing the required label. Of the 4,970 "Reconnaissance" test samples, 4,895 (98.5%) were predicted as "Port Scanning" — the model had learned the concept but used the wrong vocabulary.

This error would not be caught by:
- Standard accuracy/F1 computation (the custom script masked it)
- Training loss monitoring (the model converged normally)
- Most reproducibility checklists (they verify code is shared, not that the code is correct)

Our Perfect Score Suspicion Protocol would catch it at Level 2, Step 1 (strict vs. normalized F1 comparison).

---

## 2. Background and Related Work

### 2.1 The Reproducibility Crisis in Machine Learning

Baker (2016) documented that 70% of researchers failed to reproduce others' experiments. In machine learning specifically, Bouthillier et al. (2019) showed that random seeds alone could swing neural network results by several percentage points. For LLMs, the problem is compounded by non-deterministic generation, prompt sensitivity, and the complexity of modern training pipelines.

### 2.2 Existing Reproducibility Guidelines

**NeurIPS Reproducibility Checklist** requires code, data, and hyperparameter disclosure but does not address LLM-specific issues such as label aliasing or hallucination. **ML Reproducibility Challenge** (Pineau et al., 2021) focuses on reproducing published results but provides no guidance on verification of anomalously high scores. **HELM** (Liang et al., 2022) provides a comprehensive evaluation framework for general LLM benchmarks but is not designed for domain-specific fine-tuning evaluation.

### 2.3 LLM-Specific Evaluation Challenges

Three challenges are unique to LLM evaluation:

1. **Label aliasing**: LLMs generate free-text outputs that may match the correct concept but not the exact label string. "Port Scanning" vs. "Reconnaissance" represents correct semantic understanding with incorrect label compliance.

2. **Hallucinated labels**: LLMs can generate labels that exist in their pre-training corpus but not in the task's label vocabulary. This is distinct from random errors — the hallucinated labels are semantically meaningful but off-schema.

3. **Prompt sensitivity**: Minor changes to the instruction prompt can dramatically affect both the format and content of outputs, making evaluation fragile.

### 2.4 Information-Theoretic Task Complexity

Shannon entropy H(Y) measures the inherent complexity of a classification task. We utilize this in our checklist (Item 1) as a diagnostic for whether high performance is expected or suspicious:

$$H(Y) = -\sum_{k=1}^{K} p(y_k) \log_2 p(y_k)$$

Tasks with H(Y) < 1 bit are often trivially solvable by simple classifiers, making 100% LLM F1 expected rather than impressive.

---

## 3. The Reproducibility Checklist

We organize 30 items into 9 categories. Each item includes a pass criterion and is linked to ≥1 case study demonstrating a real failure mode.

### 3.1 Dataset Integrity (Items 1–5)

**Why**: The dataset determines the ceiling of any method. If the dataset is trivially separable, reporting high F1 is misleading without this context.

**Item 1: Train/test overlap detection.** Verify that zero training samples appear in the test set, using both exact string matching and near-duplicate detection (e.g., MinHash). *Pass criterion*: 0 overlapping samples. *Failure mode*: Our original SALAD split had train-test overlap, causing rank ablation results to be entirely spurious (Case Study 2).

**Item 2: Class distribution reporting.** Report per-class sample counts and compute label entropy H(Y). *Pass criterion*: Distribution table appears in the paper. *Failure mode*: Our SALAD classification task was 99.9% Malicious — F1 = 100% for all methods including a majority-class baseline (Case Study 5).

**Item 3: Unique pattern analysis.** Count unique input patterns versus total samples. *Pass criterion*: Report unique pattern count. *Failure mode*: SALAD has only 87 unique input patterns (each repeated ~113 times), making it a lookup-table task rather than a generalization challenge.

**Item 4: Zero-ambiguity check.** Report whether each unique input maps to exactly one label. *Pass criterion*: Report ambiguity rate. *Failure mode*: SALAD has 0% ambiguity — every pattern maps to exactly one attack category. K-Means ARI = 0.91, meaning unsupervised clustering nearly equals supervised classification.

**Item 5: Feature importance analysis.** Compute mutual information between each feature and the target label. *Pass criterion*: Report MI per feature. *Failure mode*: Network Segment feature in SALAD has MI = 0.000 — it provides zero information but is still included in the input.

### 3.2 Baseline Selection (Items 6–9)

**Why**: Without baselines, there is no way to assess whether LLM fine-tuning adds value over cheaper methods.

**Item 6: Traditional ML baseline.** Include at least one of Decision Tree, SVM, or Logistic Regression with TF-IDF features. *Pass criterion*: Traditional ML F1 reported on same test set. *Failure mode*: Our DT achieved 87.4% on Attack Category — the LLM adds only 12.6% (strict) at 100× the training cost (Case Study 3).

**Item 7: Pre-trained transformer baseline.** Include BERT-base or equivalent. *Pass criterion*: BERT F1 reported. *Failure mode*: BERT scored 81.4% on SALAD, lower than SVM (90.9%), challenging the assumption that deep learning always wins.

**Item 8: In-context learning comparison.** Include at least one commercial LLM via API. *Pass criterion*: ICL F1 reported, or justified exclusion (cost, privacy). *Failure mode*: ICL may match fine-tuning at lower total cost for small evaluation volumes.

**Item 9: Matched evaluation conditions.** All methods evaluated on the same test set, with the same metrics, using the same data splits. *Pass criterion*: Cross-method evaluation table with consistent setup. *Failure mode*: Comparing models evaluated on different test sets produces meaningless rankings.

### 3.3 Metric Selection (Items 10–13)

**Item 10: Macro-F1 for imbalanced data.** Report macro-averaged F1, not accuracy. *Pass criterion*: Macro-F1 as primary metric. *Failure mode*: 99.9% accuracy = always predicting the majority class.

**Item 11: Per-class F1 breakdown.** Report F1 for every class, including rare classes. *Pass criterion*: Per-class table in paper. *Failure mode*: Mistral's aggregate F1 was 91.7%, but Analysis class was 0% and Backdoor was 7.5% (Case Study 4).

**Item 12: Strict AND normalized F1.** Report both metrics prominently, with strict F1 as the primary metric. *Pass criterion*: Both metrics appear, strict is labeled "primary." *Failure mode*: Our 44.3% gap between strict (55.7%) and normalized (100%) would have gone undetected with only normalized reporting (Case Study 1).

**Item 13: Random and majority baselines.** Compute F1 for random proportional and majority-class predictions. *Pass criterion*: Both baselines reported. *Failure mode*: Random Atk F1 = 12.5% (confirming the model is significantly above chance), but random Classification accuracy = 99.9% (confirming the metric is uninformative).

### 3.4 Statistical Rigor (Items 14–16)

**Item 14: Multi-seed (≥3 runs).** Report mean and standard deviation across at least 3 random seeds. *Pass criterion*: Mean ± std for main results.

**Item 15: Significance testing.** Apply paired t-test or Wilcoxon signed-rank test for key comparisons. *Pass criterion*: p-values reported for claimed improvements.

**Item 16: Confidence intervals.** Report 95% confidence intervals for main results. *Pass criterion*: CI appears in results tables.

### 3.5 Label Quality and Hallucination Audit (Items 17–20)

**Why**: This category addresses the most LLM-specific evaluation challenge — models that understand the task perfectly but produce incorrect label strings.

**Item 17: Predicted vocabulary audit.** Compare the set of labels the model produces against the set of ground-truth labels. *Pass criterion*: Predicted vocabulary listed explicitly. *Failure mode*: Our 0.8B model produced 12 unique labels when only 8 exist, including "Port Scanning," "Backdoors," "Bots," and "L2TP."

**Item 18: Hallucinated label inventory.** For each predicted label not in the true vocabulary, report: (a) count, (b) which true label it replaces, (c) type (sub-category, synonym, typo, or unrelated). *Pass criterion*: Complete inventory table. *Failure mode*: "Port Scanning" appeared 4,895 times, replacing "Reconnaissance" — a MITRE ATT&CK sub-technique bleeding from pre-training.

**Item 19: Alias mapping with justification.** If using label normalization, provide explicit justification for each alias. *Pass criterion*: Alias table with rationale. *Rule of thumb*: if a reviewer cannot verify the alias validity in under 30 seconds, the alias is not justified.

**Item 20: Normalized F1 gap rule.** If the gap between normalized and strict F1 exceeds 10%, treat this as a red flag requiring prominent disclosure. *Pass criterion*: Gap < 10%, or gap prominently discussed. *Failure mode*: A 44.3% gap was hidden inside a helper function.

### 3.6 Ablation Design (Items 21–23)

**Item 21: Hyperparameter sensitivity.** Test at least 3 values for key hyperparameters. *Pass criterion*: Sensitivity table or plot.

**Item 22: Training data scaling.** Evaluate at least 3 training set sizes. *Pass criterion*: Scaling curve reported. *Failure mode*: Our scaling curve was flat for normalized F1 (1K-20K all scored 100%) but revealed a real trend in strict F1 (77.8% → 100%).

**Item 23: Confound isolation.** If any ablation results are surprising, re-run on verified clean data. *Pass criterion*: Clean re-run reported for anomalous results. *Failure mode*: LoRA rank 32 scored 66% on leaky data but 100% on clean data — the entire "rank anomaly" was a data leakage artifact (Case Study 2).

### 3.7 Cost Transparency (Items 24–26)

**Item 24: GPU hours per experiment.** *Pass criterion*: Training time reported.

**Item 25: API cost for ICL experiments.** *Pass criterion*: Per-sample or total cost reported.

**Item 26: Cost-per-improvement.** Report ΔF1 per dollar spent. *Pass criterion*: Cost-performance ratio discussed. *Example*: SVM→LLM on SALAD: +12.6% strict F1 for $2.24 training cost = 5.6% per dollar.

### 3.8 Reproducibility Artifacts (Items 27–28)

**Item 27: Code and configurations public.** *Pass criterion*: Repository with training scripts, evaluation scripts, and data processing code.

**Item 28: Environment documentation.** *Pass criterion*: CUDA version, framework version, hardware specification (GPU model, VRAM), and random seeds.

### 3.9 Perfect Score Suspicion Protocol (Items 29–30)

**Why**: When a metric reports ≥99%, the default response should be skepticism, not celebration. Our protocol provides a structured verification procedure.

**Item 29: Level 1-2 verification.** When any metric ≥ 99%, pass all 8 quick checks within 35 minutes. *Pass criterion*: All 8 checks documented.

**Item 30: Level 3-4 evidence.** For submitted papers, complete deep inspection and archive evidence. *Pass criterion*: Evidence trail saved.

The full 4-level protocol contains 15 individual checks and is detailed in Appendix A.

---

## 4. Case Studies

### Case Study 1: Label Aliasing Inflates F1 by 44.3%

**Context**: Qwen3.5-0.8B fine-tuned with QLoRA (rank 64) on 5,000 SALAD training samples, evaluated on 9,851 test samples for 8-class attack category classification.

**Observed**: Custom evaluation script reports Attack Category Macro-F1 = 100.0%.

**Investigation**: Manual spot-check of 15 random predictions revealed only 3/15 exact matches. sklearn.metrics.f1_score without normalization returned 55.7%. The custom script contained ATTACK_ALIASES mapping "Port Scanning" → "Reconnaissance" (4,895 samples), "Backdoors" → "Backdoor" (20 samples), "Bots" → "Backdoor" (6 samples), and "L2TP" → "Reconnaissance" (4 samples).

**Root cause**: The model learned the correct semantic mapping from its pre-training corpus (MITRE ATT&CK documentation uses "Port Scanning" as technique T1046 under "Reconnaissance" tactic TA0043) but did not learn to use the SALAD-specific parent category label.

**Impact**: Without detection, this would have been published as "100% F1" — misleading readers about the model's actual label compliance.

**Checklist items that would catch this**: Items 12, 17, 18, 29 (L2.1), 29 (L2.3).

### Case Study 2: Data Leakage Produces Spurious Ablation Patterns

**Context**: LoRA rank ablation (ranks 16, 32, 64, 128) on Qwen3.5-0.8B.

**On leaky data**: Rank 32 = 66.1%, Rank 128 = 62.3% — conclusion: "higher rank causes overfitting."

**On clean data**: Rank 32 = 100%, Rank 128 = 95.8% — conclusion: "rank has minimal effect; only rank 128 shows mild hallucination."

**Root cause**: The original data split had train-test overlap. Higher-rank adapters had more capacity to memorize specific leaked patterns, producing a spurious inverse relationship between rank and performance.

**Impact**: The "overfitting" conclusion was entirely an artifact of data leakage. The clean-data results tell a completely different story.

**Checklist items that would catch this**: Items 1, 23.

### Case Study 3: Traditional ML Outperforms LLMs

**Context**: 6-domain comparison spanning H(Y) = 0.85 to 6.16 bits.

**Finding**: On SALAD Attack Category (H=1.24), SVM achieves 90.9% F1 with TF-IDF features at zero training cost. The fine-tuned LLM adds only 9.1% (normalized) or less (strict) at $2.24 training cost. BERT-base scores 81.4% — lower than SVM.

**Implication**: For low-entropy tasks, claiming LLM superiority without SVM comparison is misleading.

**Checklist items that would catch this**: Items 6, 7, 29 (L1.3).

### Case Study 4: Catastrophic Failure on Rare Classes

**Context**: Mistral-7B-v0.3 fine-tuned on SALAD 5K.

**Observed**: Aggregate Attack Category F1 = 91.7% (appears acceptable). Per-class: Analysis (68 samples) = 0.000 F1, Backdoor (51 samples) = 0.075 F1.

**Root cause**: Mistral hallucinated "Port Scanning" for Analysis and "Shellcode" for Backdoor — sub-category labels from its English-centric MITRE ATT&CK knowledge that overpowered fine-tuning on rare classes.

**Impact**: A deployed model achieving 91.7% aggregate F1 would completely fail on two attack categories, potentially missing critical security incidents.

**Checklist items that would catch this**: Items 11, 17, 18.

### Case Study 5: Meaningless Binary Classification

**Context**: SALAD Classification task (Malicious vs Benign).

**Finding**: 99.9% of test samples are Malicious. All models — including a trivial majority-class predictor — achieve F1 ≈ 100%. Classification F1 carries zero discriminative value.

**Implication**: Reporting "100% Classification F1" without disclosing the 99.9% class imbalance is misleading.

**Checklist items that would catch this**: Items 2, 13, 29 (L1.2).

### Case Study 6: Eval Script Tests on Wrong Dataset

**Context**: Zero-shot and cross-domain experiments required per-category and domain-specific test sets. 26 evaluation jobs were submitted.

**Observed**: All 26 jobs reported "COMPLETED" with exit code 0, but produced zero meaningful predictions. Zero-shot models trained without "Analysis" were tested on the full SALAD test set instead of the held-out "Analysis" samples. Cross-domain models trained on AG News were also tested on SALAD.

**Root cause**: The shared eval script (`eval.sh`) hardcoded `--eval_dataset clean_test`. Researchers assumed the script accepted a custom test dataset parameter, but it did not. The fix required creating `flexible_eval.sh` that takes the test dataset as an explicit argument.

**Impact**: 26 GPU-hours wasted (13 jobs × 2h each). Results appeared complete (exit code 0) but were scientifically meaningless — testing a zero-shot model on the wrong test set produces no useful signal.

**Checklist items that would catch this**: Item 9 (matched conditions), Item 28 (script documentation). A simple pre-flight check — "does this script test on the correct dataset?" — would have prevented the waste.

**New Item 9b: Pre-flight eval verification.** Before submitting batch eval jobs, verify on a single sample that the eval script uses the correct test dataset. *Pass criterion*: First prediction matches expected domain. *Rule*: Always parameterize test datasets; never hardcode them in shared scripts.

---

## 5. Discussion

### 5.1 Cost of Compliance

Implementing the full 30-item checklist adds approximately 4–8 hours per paper. The Perfect Score Suspicion Protocol (Items 29–30) requires 35 minutes for Level 1–2 and 2–4 hours for Level 3–4. We argue this investment is justified: publishing incorrect results causes far greater downstream costs, including retraction risk, wasted follow-up research by other groups, and reputational damage.

### 5.2 When to Relax Requirements

Not all items are equally critical for all contexts. Item 8 (ICL comparison) may be omitted when API access is restricted or cost-prohibitive. Item 15 (significance testing) may be relaxed for differences exceeding 10%. Level 4 of the Perfect Score Protocol (cross-domain validation) may be omitted for papers that are inherently domain-specific.

### 5.3 The Strict vs. Normalized Debate

We encountered significant internal discussion about whether to report "Port Scanning" → "Reconnaissance" as a correct prediction. Arguments for normalization: the model understands the concept correctly; MITRE ATT&CK defines Port Scanning as a sub-technique of Reconnaissance. Arguments against: the model violated the label schema specified in the prompt; the prediction is objectively wrong by the task definition.

Our recommendation: **strict F1 = primary metric, normalized F1 = supplementary metric with explicit alias documentation**. Researchers must never report only the normalized score.

### 5.4 Domain-Specific Considerations

**Cybersecurity**: The MITRE ATT&CK framework creates a natural label hierarchy that induces aliasing. Models pre-trained on cybersecurity text are especially prone to hallucinating sub-technique names.

**Clinical NLP**: Patient privacy constraints may require synthetic data release for reproducibility (Item 27). Medical coding systems (ICD-10) create similar aliasing risks.

**Legal NLP**: Long document classification requires document-level feature importance analysis (Item 5) rather than token-level measures.

---

## 6. Conclusion

We present the first comprehensive evaluation checklist designed for LLM experiments in applied domains. Our 9-category, 30-item framework addresses critical gaps in existing reproducibility guidelines, particularly in label quality verification, hallucination auditing, and high-score suspicion protocols. The checklist is validated against 5 real case studies demonstrating that current evaluation practices can mask F1 inflation of up to 44.3%, produce entirely spurious ablation conclusions, and generate meaningless metrics on degenerate classification tasks.

We urge the community to adopt structured verification procedures — especially the Perfect Score Suspicion Protocol — as mandatory pre-submission gates. The 35-minute investment for Level 1–2 verification can prevent the publication of seriously misleading results.

The full checklist, executable validation scripts, and case study data are available at [repository URL].

---

## Acknowledgments

The authors acknowledge the NSTDA Supercomputer Center (ThaiSC) for providing computing resources on the Lanta HPC system.

---

## Appendix A: Perfect Score Suspicion Protocol (Complete)

[See workflow: /perfect-score-check.md]

## Appendix B: Executable Validation Scripts

[See workflow: /perfect-score-check.md — all scripts are directly runnable]

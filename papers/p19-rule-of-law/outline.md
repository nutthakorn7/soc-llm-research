# P19: Beyond Accuracy — A Reproducibility Checklist for LLM Evaluation in Applied Domains

## Paper Outline

### Abstract
We propose a 30-item reproducibility checklist for LLM experiments in applied domains, derived from our experience conducting 17 research papers across cybersecurity, NLP, and legal text classification. We identify 8 categories of common evaluation pitfalls and provide concrete guidelines with pass/fail criteria.

### 1. Introduction
- Reproducibility crisis in ML (Baker, 2016)
- LLM evaluation is especially challenging: non-deterministic outputs, label aliasing, prompt sensitivity
- Existing checklists (NeurIPS, ML Reproducibility) don't address LLM-specific issues

### 2. Methodology
- **Source**: 17 papers, 34 models, 4 domains, 50+ experiments
- **Process**: Document every evaluation failure and root cause
- **Output**: 30-item checklist organized in 8 categories

### 3. The Checklist (8 Categories)

#### 3.1 Dataset Integrity
- Train/test overlap detection
- Class distribution reporting
- Entropy measurement

#### 3.2 Baseline Selection
- Traditional ML (DT, SVM) mandatory
- Pre-trained transformer (BERT) mandatory
- ICL comparison mandatory
- Match evaluation conditions

#### 3.3 Metric Selection
- Macro-F1 for imbalanced data
- Per-class breakdown
- Strict vs relaxed matching

#### 3.4 Statistical Rigor
- Multi-seed (≥3)
- Significance testing
- Confidence intervals

#### 3.5 Label Quality
- **NEW: Label aliasing detection** — our key contribution
- Synonym normalization protocol
- Report both raw and normalized

#### 3.6 Ablation Design
- Hyperparameter sensitivity
- Training data scaling

#### 3.7 Cost Transparency
- GPU hours
- API cost
- Inference latency

#### 3.8 Reproducibility Artifacts
- Code, data, configs public
- Hardware specs
- Environment documentation

### 4. Case Studies
- **Case 1**: Label aliasing caused 54% F1 drop (Mistral on SALAD)
- **Case 2**: SVM outperforms 3 LLMs on low-entropy tasks
- **Case 3**: Rank 128 LoRA worse than Rank 16 (overfitting)

### 5. Discussion
- Cost of compliance vs benefit
- When to relax requirements
- Domain-specific adaptations

### 6. Conclusion

## Target
- **ACM Computing Surveys** (Q1, IF 16.6)
- 20-25 pages
- No experiments needed — meta-research paper

# SOC-FT: LLM Fine-tuning for SOC Alert Triage

## Project Overview
Fine-tune open-weight LLMs on the SALAD dataset using LlamaFactory on NSTDA Lanta HPC. 
**Paper angle**: "Task-Complexity Analysis: When Do LLMs Outperform Traditional ML for Security Operations?"

Key findings: SALAD has **870 unique alert patterns** (7 benign, 863 malicious) from 1.9M records. Decision Tree achieves 100% on Classification/Triage but only **87.4%** on Attack Category → LLMs needed for complex multi-class tasks.

## HPC Environment
> ⚠️ **LANTA HPC IS NO LONGER AVAILABLE (as of Mar 2026)**. Use **Vast.ai** for GPU training.

- **Cluster**: NSTDA Lanta (8.15 PFlop/s, 704× NVIDIA A100) — **DECOMMISSIONED**
- **Alternative**: Vast.ai cloud GPU (A100/H100 instances)
- **Account**: `lm2002` / Project `lt200473-ttctvs` (historical)
- **SSH**: `ssh lanta` — **NO LONGER WORKS**
- **Conda**: `module load cuda/12.6 Mamba/23.11.0-0 && source activate soc-finetune`

## Directory Layout

### Local (Mac)
```
/Users/pop7/Code/Lanta/
├── CLAUDE.md
├── scripts/
│   ├── train.sh                # Qwen3.5-9B 50K (4 GPUs)
│   ├── eval.sh                 # Eval: predict on clean_test (4h limit)
│   ├── fast_eval.sh            # ⚡ Faster eval (batch=8, tokens=150)
│   ├── baselines.py            # DT/RF/GBM/SVM baselines
│   ├── calc_f1.py              # Per-task F1 from predictions
│   ├── sanity_check.py         # 5-agent reviewer suite
│   ├── orchestrator.sh         # Auto: training→eval→F1 pipeline
│   ├── sync_and_f1.sh          # One-command: sync + calc all F1
│   ├── cascade_v2.py           # Fixed DT→LLM cascade + vLLM latency
│   ├── zero_shot_transfer.py   # P18: Leave-one-category-out
│   ├── cross_domain_analysis.py# General AI: entropy across 4 domains
│   ├── llm_eval_audit.py       # P19: Automated 12/30 checklist tool
│   ├── generate_all_figures.py # All 42 paper figure PDFs (matplotlib)
│   ├── generate_figures.py     # Paper-ready HTML charts + tables
│   ├── generate_latex.py       # 5 LaTeX tables from analysis
│   ├── generate_model_card.py  # Responsible AI model card
│   ├── clustering_analysis.py  # K-Means/DBSCAN + MI features
│   ├── adversarial_test.py     # Perturbation robustness
│   ├── training_cost.py        # GPU hours, cost, CO₂
│   └── train_*.sh              # Per-model training scripts
├── papers/                     # 15 LaTeX papers (all ≥6 pages)
│   ├── p3-soc-ft/              # Flagship: 20p, 23T, 40R ⭐⭐⭐
│   ├── p5-cascade/             # 7p, 8T, 22R ⭐⭐
│   ├── p6-scaling/             # 9p, 11T, 33R ⭐⭐⭐
│   ├── p7-cost-efficient/      # 8p, 10T, 24R ⭐⭐
│   ├── p8-task-complexity/     # 7p, 7T, 28R ⭐⭐⭐
│   ├── p9-rlhf-dpo/           # 6p, 7T, 20R ⭐⭐ (negative result)
│   ├── p14-oft-vs-lora/       # 7p, 12T, 20R ⭐⭐
│   ├── p15-multi-task/         # 7p, 8T, 22R ⭐⭐
│   ├── p18-zero-shot/          # 7p, 8T, 22R ⭐⭐
│   ├── p19-rule-of-law/        # 9p, 9T, 26R ⭐⭐⭐ (+audit tool)
│   ├── p20-general-ai/         # 7p, 10T, 21R ⭐⭐
│   ├── p21-sub-1b/             # 8p, 9T, 25R ⭐⭐
│   ├── p22-lora-rank/          # 7p, 8T, 28R ⭐⭐⭐
│   ├── p23-edge-quant/         # 7p, 9T, 21R ⭐⭐
│   └── p24-cyber-datasets/     # 10p, 9T, 29R ⭐⭐
└── results/
    ├── paper_results/          # Analysis JSONs + figures
    └── general_ai/             # Cross-domain datasets + results
```

### Lanta HPC
```
/project/lt200473-ttctvs/soc-finetune/
├── data/
│   ├── train_*_clean.json     # Deduped, no leakage (1K/5K/10K/20K/50K)
│   ├── val_held_out.json      # 9,368 val (stratified, both classes)
│   ├── test_held_out.json     # 9,851 test (stratified, both classes)
│   ├── dataset_info.json      # LlamaFactory dataset registry
│   └── train_full.json        # Original 1.9M (DO NOT use directly)
├── models/                    # 11 pre-downloaded models (+ BERT)
├── scripts/                   # Uploaded scripts (mirrored from local)
└── outputs/
    ├── clean-qwen35-*/        # Training adapters
    ├── eval-*/                # Eval predictions + F1
    ├── paper_results/         # Deployment, cascade, latency, etc.
    └── orchestrator.log
```

### LlamaFactory
```
/project/lt200473-ttctvs/workshop-pretrain/LlamaFactory/
```

## Models (11 LLMs + BERT)

| Model | Size | Template | Status |
|-------|------|----------|--------|
| Qwen3.5-9B | 18 GB | `qwen3_5` | ✅ Primary |
| Qwen3.5-0.8B | 1.6 GB | `qwen3_5` | ✅ Edge |
| Qwen3-8B | 16 GB | `qwen3` | ✅ |
| Mistral 7B v0.3 | 14 GB | `mistral` | ✅ |
| Gemma 3 4B IT | 8 GB | `gemma` | ✅ Training |
| DeepSeek-R1 7B | 15 GB | `deepseek3` | ✅ |
| Phi-4-mini 3.8B | 7.2 GB | `phi4` | ✅ |
| Granite 3.3 8B | 16 GB | `default` | ✅ Training |
| SmolLM2 1.7B | 22 GB | `default` | ✅ |
| BERT-base | 440 MB | HF Trainer | ✅ 4 domains |
| Llama 3.1 8B | — | `llama3` | ❌ No weights |

## Dataset

> ⚠️ **CRITICAL**: Use `clean_*` datasets ONLY. Original splits had data leakage!

- **Registry names**: `clean_1k`, `clean_5k`, `clean_10k`, `clean_20k`, `clean_50k`, `clean_val`, `clean_test`
- **Unique prompts**: 870 total (7 benign, 863 malicious)
- **Split (by prompt, stratified)**: 696 train / 87 val / 87 test → **0 overlap**
- **Train pool**: 1.93M samples (subsets: 1K/5K/10K/20K/50K)
- **Val**: 9,368 samples (87 prompts) | **Test**: 9,851 samples (87 prompts, 8 attack categories)
- **DT baseline**: 100% cls, 100% tri, **87.4% atk**

> ⚠️ **AUDIT (2026-03-10)**: `scale-qwen35-*` adapters were trained on OLD leaky data (`train_Xk.json`). Use `clean-qwen35-*` adapters instead (`train_Xk_clean.json`). All `mm-*`, `seed-*`, `abl-*` jobs use clean data ✅

### dataset_info.json Registry (key entries)
| Registry | File | Samples | Status |
|----------|------|---------|--------|
| `salad_val` | `val_5k.json` | 5,000 | ❌ OLD — do NOT use for eval |
| `clean_val` | `val_held_out.json` | 9,368 | ✅ Clean val |
| `clean_test` | `test_held_out.json` | 9,851 | ✅ Clean test (use for all evals) |

## Training Config
- **Method**: QLoRA (4-bit quantization via bitsandbytes)
- **LoRA**: rank 64, alpha 128, target all linear layers
- **LR**: 2e-4 cosine with 10% warmup
- **Epochs**: 3
- **Key flag**: `--quantization_method bnb` (NOT `bitsandbytes`)

## Paper Portfolio (Mar 13, 2026 — ALL COMPILED ✅ 96 PAGES)

| # | Paper | Pages | Tables | Figs | Refs | P19 | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **P3** | Mind the Label Gap (IEEE TDSC) | 14 | 23 | 4 | 40 | 29/30 | ⭐⭐⭐ Flagship |
| **P5** | Entropy-Aware Cascade (FGCS) | 6 | 8 | 3 | 22 | 25/30 | ⭐⭐ |
| **P6** | 1K Labels Is All You Need (ESWA) | 8 | 11 | 4 | 33 | 28/30 | ⭐⭐⭐ |
| **P7** | $0.60 Is All You Need (IEEE Access) | 6 | 10 | 4 | 24 | 26/30 | ⭐⭐ |
| **P8** | Entropy Predicts LLM (IS) | 6 | 7 | 3 | 28 | 27/30 | ⭐⭐⭐ |
| **P9** | DPO Destroys Classification (TMLR) | 5 | 7 | 2 | 21 | 21/30 | ⭐⭐ |
| **P14** | LoRA vs. OFT (PR) | 5 | 10 | 2 | 22 | 25/30 | ⭐⭐ |
| **P15** | One Model, Three Tasks (ASC) | 6 | 8 | 3 | 22 | 25/30 | ⭐⭐ |
| **P18** | Zero-Shot Generalization (TKDD) | 5 | 8 | 2 | 22 | 22/30 | ⭐⭐ |
| **P19** | Reproducibility Checklist (C&S) | 7 | 9 | 3 | 26 | — | ⭐⭐⭐ |
| **P20** | Cross-Domain Transfer (TKDE) | 5 | 10 | 3 | 21 | 24/30 | ⭐⭐ |
| **P21** | Sub-1B Models Follow (KBS) | 7 | 9 | 3 | 25 | 24/30 | ⭐⭐ |
| **P22** | Higher Rank, More Hallucination (NC) | 6 | 8 | 4 | 28 | 27/30 | ⭐⭐⭐ |
| **P23** | Quantize and Deploy (IoT J.) | 5 | 9 | 2 | 21 | 23/30 | ⭐⭐ |
| **P24** | Cyber Dataset Analysis | 7 | 9 | 2 | 29 | 24/30 | ⭐⭐ |

### P19 Self-Audit Results (14 papers audited)
- Mean score: **25.0/30** (83.3%)
- Most common failure: **Single seed** (Items 14–16)
- Multi-seed papers (P3,P6,P22,P15) score ≥25/30
- Automated tool `llm_eval_audit.py` tested ✅
- Figure generation: `generate_all_figures.py` produces all 42 PDFs

## Paper Finalization Workflow
> Run `/paper-finalize` on **every paper** before submission!
> 4 rounds: Proofread → Plagiarism check → Humanize → Verification
> See `.agents/workflows/paper-finalize.md`

## Workflow
```bash
# Upload script
ssh lanta "cat > /project/.../scripts/script.sh" < scripts/script.sh

# Submit training
ssh lanta 'cd /project/lt200473-ttctvs/soc-finetune && sbatch scripts/train.sh'

# Monitor all jobs
ssh lanta 'squeue -u lm2002'

# Run P19 automated audit
python3 scripts/llm_eval_audit.py --train train.jsonl --test test.jsonl --preds preds.jsonl --labels labels.txt

# Sync results
rsync -avz --exclude='*.safetensors' --exclude='checkpoint-*' lanta:/project/.../outputs/ results/
```

## Known Issues
- GPU nodes have no internet → always use local model paths
- `quantization_method` must be `bnb`, not `bitsandbytes`
- Qwen3.5 uses `qwen3_5` template (not `qwen` or `qwen3`)
- **SALAD has only 870 unique patterns** — scaling law uses unique prompts, not raw count
- **Original data splits had leakage** — ALWAYS use `clean_*` datasets
- **`scale-qwen35-*` adapters are INVALID** — use `clean-qwen35-*` instead
- **`eval-final-q35-5k` result is INVALID** — double leakage
- **test_held_out.json has 8/15 attack categories** (87 test prompts)
- **torchrun + QLoRA (4-bit) = CRASH** — DDP incompatible with BnB quantization. Use 1 GPU for QLoRA
- **LlamaFactory does NOT support `finetuning_type: dora`** — only `lora`, `oft`, `freeze`, `full`
- **cascade_results.json v1 was WRONG** — DT confidence always 1.0 → llm_calls=0. Fixed in `cascade_v2.py`
- **LlamaFactory batch gen is slow** (~2h for 9B eval). Use `fast_eval.sh` (batch=8, tokens=150) for 2-3× speedup
- LlamaFactory predict requires: `jieba`, `nltk`, `rouge_chinese`, `rouge_score`, `sacrebleu`
- **`wc -l` shows 0** for single-line JSON — use `python3 -c "len(json.load(...))"` instead

## Acknowledgment Template (ใส่ทุก paper ที่ใช้ Lanta)
> The authors acknowledge the NSTDA Supercomputer Center (ThaiSC) and the National Science and Technology Development Agency (NSTDA), National e-Science Infrastructure Consortium, Ministry of Higher Education, Science, Research and Innovation (MHESI), Thailand, for providing the LANTA High-Performance Computing (HPC) system (8.15 PFlop/s, HPE Cray EX, 704 NVIDIA A100 GPUs) that has contributed to the research results reported within this paper.

**Papers ที่ต้องใส่**: P3, P5, P6, P7, P14, P15, P18, P20, P21, P22, P23

## Q1 Publication Checklist
ก่อน submit ทุก paper → ตรวจ `q1_rule_of_law.md` ให้ครบทุกข้อ:
- ≥2 datasets, ≥4 baselines (DT, SVM, BERT, ICL)
- Macro-F1 + per-class, 3-5 seeds (mean±std)
- Statistical test (Wilcoxon/McNemar p<0.05)
- BERT baseline, cost analysis, code public
- **Run `/paper-finalize` workflow** (proofread + plagiarism + humanize)
- **Run `llm_eval_audit.py`** on predictions before reporting results

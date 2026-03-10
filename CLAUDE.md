# SOC-FT: LLM Fine-tuning for SOC Alert Triage

## Project Overview
Fine-tune open-weight LLMs on the SALAD dataset using LlamaFactory on NSTDA Lanta HPC. 
**Paper angle**: "Task-Complexity Analysis: When Do LLMs Outperform Traditional ML for Security Operations?"

Key findings: SALAD has **870 unique alert patterns** (7 benign, 863 malicious) from 1.9M records. Decision Tree achieves 100% on Classification/Triage but only **87.4%** on Attack Category → LLMs needed for complex multi-class tasks.

## HPC Environment
- **Cluster**: NSTDA Lanta (8.15 PFlop/s, 704× NVIDIA A100)
- **Account**: `lm2002` / Project `lt200473-ttctvs`
- **SSH**: `ssh lanta` (passwordless via publickey)
- **GPU nodes have NO internet** — models must be pre-downloaded
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
│   ├── generate_figures.py     # Paper-ready HTML charts + tables
│   ├── generate_latex.py       # 5 LaTeX tables from analysis
│   ├── generate_model_card.py  # Responsible AI model card
│   ├── clustering_analysis.py  # K-Means/DBSCAN + MI features
│   ├── adversarial_test.py     # Perturbation robustness
│   ├── training_cost.py        # GPU hours, cost, CO₂
│   └── train_*.sh              # Per-model training scripts
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
├── models/                    # 10 pre-downloaded models
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

## Models (10 downloaded)

| Model | Size | Template | Status |
|-------|------|----------|--------|
| Qwen3.5-9B | 18 GB | `qwen3_5` | ✅ Primary |
| Qwen3.5-0.8B | 1.6 GB | `qwen3_5` | ✅ Edge |
| Qwen3-8B | 16 GB | `qwen3` | ✅ |
| Mistral 7B v0.3 | 14 GB | `mistral` | ✅ |
| Gemma 3 4B IT | 8 GB | `gemma` | ✅ |
| DeepSeek-R1 7B | 15 GB | `deepseek` | ✅ |
| Phi-4-mini 3.8B | 7.2 GB | `phi4` | ✅ |
| Granite 3.3 8B | 16 GB | `granite` | ✅ |
| SmolLM2 1.7B | 22 GB | `smollm` | ✅ |
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

## Experiments (22+ total)

### P3 SOC-FT: Core (14 eval RUNNING — 18:48 Mar 10)
1. **Scaling law**: 1K/5K/10K/20K/50K → Qwen3.5-9B
   - 1K/5K/10K trained ✅, eval RUNNING
   - 20K: training RUNNING (63%), 50K: training RUNNING (1 GPU)
2. **Multi-seed**: seeds 42/123/2024 (trained ✅, eval RUNNING)
3. **Multi-model**: 9 models × 5K (trained ✅, eval RUNNING)
4. **Ablation**: rank 16/32/128 + LR 1e-4/5e-4 (trained ✅, eval RUNNING)

### Analysis (all done ✅)
5-14. Baselines, task complexity, cost-benefit, deployment, cascade, clustering, feature importance, adversarial, training cost, paper figures

### P18 Zero-Shot Transfer (8 jobs RUNNING)
15. Leave-one-category-out × 8 folds (0.8B, training RUNNING)
16. DT baseline: **0% accuracy** on all 8 categories

### General AI Paper (3 jobs RUNNING)
17. Cross-domain entropy analysis: AG News, GoEmotions, LEDGAR
18. DT baselines: 55.6%, 21.7%, 24.3% (entropy 2.0, 3.7, 6.2)

### P4 Survey (planning)
19. SLR outline + 45 references collected

## Workflow
```bash
# Upload script
ssh lanta "cat > /project/.../scripts/script.sh" < scripts/script.sh

# Submit training
ssh lanta 'cd /project/lt200473-ttctvs/soc-finetune && sbatch scripts/train.sh'

# Monitor all jobs
ssh lanta 'squeue -u lm2002'

# Auto-submit evals for completed training
ssh lanta 'bash /project/.../scripts/auto_eval_chain.sh'

# Run full master evaluation
ssh lanta 'python3 /project/.../scripts/master_eval.py /project/.../soc-finetune'

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

**Papers ที่ต้องใส่**: P3, P5, P6, P7, P14, P15, P18, General AI

## Q1 Publication Checklist
ก่อน submit ทุก paper → ตรวจ `q1_rule_of_law.md` ให้ครบทุกข้อ:
- ≥2 datasets, ≥4 baselines (DT, SVM, BERT, ICL)
- Macro-F1 + per-class, 3-5 seeds (mean±std)
- Statistical test (Wilcoxon/McNemar p<0.05)
- BERT baseline, cost analysis, code public

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
│   ├── train.sh                # Qwen3.5-9B 50K (4 GPUs) [Lanta]
│   ├── kaggle_q1_batch1.py     # ⭐ Kaggle: seeds + P8 (autosave)
│   ├── vast_q1.sh              # Vast.ai: P8 LedGAR + P9 DPO β
│   ├── vast_remaining.py       # ⭐ P8 LedGAR + P6 + P9 DPO fix
│   ├── vast_orpo.py            # ⭐ P9 ORPO λ sweep (manual impl)
│   ├── vast_p6_9b.py           # ⭐ P6 7B QLoRA scaling
│   ├── vast_v2_extra.py        # P14 OFT + P9 7B
│   ├── download_all_vastai.sh  # ⭐ One-shot download from all machines
│   ├── eval.sh                 # Eval: predict on clean_test (4h limit)
│   ├── fast_eval.sh            # ⚡ Faster eval (batch=8, tokens=150)
│   ├── baselines.py            # DT/RF/GBM/SVM baselines
│   ├── calc_f1.py              # Per-task F1 from predictions
│   ├── sanity_check.py         # 5-agent reviewer suite
│   ├── train_crossdomain.py    # Cross-domain QLoRA (PyTorch loop)
│   ├── zero_shot_transfer.py   # P18: Leave-one-category-out
│   ├── cross_domain_analysis.py# General AI: entropy across 4 domains
│   ├── llm_eval_audit.py       # P19: Automated 12/30 checklist tool
│   ├── generate_all_figures.py # All 42 paper figure PDFs (matplotlib)
│   ├── training_cost.py        # GPU hours, cost, CO₂
│   └── train_*.sh              # Per-model training scripts
├── papers/                     # 16 LaTeX papers (all ≥6 pages)
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
└── results/                    # ⭐ 200+ files (see results/README.md)
    ├── master_results.csv      # ALL results consolidated
    ├── kaggle-mar16/            # P8 AG News/GoEmotions, P20/P21
    ├── vastai-priority1/        # P14 LoRA ×10
    ├── vastai-priority2/        # P15, P22, P23
    ├── vastai-mar16-live/       # P9 DPO, P18 LOO, P15
    ├── vastai-v2-final/         # P14 OFT ×10, P9 7B ×3
    ├── vastai-remaining/        # P8 LedGAR, P6 0.5B
    ├── vastai-dpo-partial/      # P9 DPO 4β ×3 seeds (12 runs)
    ├── vastai-orpo-partial/     # P9 ORPO 3λ ×5 seeds (15 runs)
    ├── vastai-p6-7b-partial/    # P6 7B 5 seeds
    └── README.md                # Folder guide + summary table
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

## Paper Portfolio (Mar 17, 2026 — ALL EXPERIMENTS COMPLETE ✅)

| # | Title | Key Result | Status |
|---|---|---|---|
| **P3** | Mind the Label Gap: Semantic Hallucination in Fine-Tuned LLMs for SOC | — | ✅ Ready |
| **P5** | Entropy-Driven Cascade Routing | — | ✅ Ready |
| **P6** | Non-Monotonic Scaling of Label Compliance | 0.5B: 0.918, 7B: 0.817±0.144 | ✅ 7B 5 seeds |
| **P7** | $0.60 per Model: Cost of Label-Compliant LLM Fine-Tuning | — | ✅ Ready |
| **P8** | Label Entropy as a Predictor of LLM Necessity | AG 0.91/GoEmo 0.42/LedGAR 0.62 | ✅ 3 datasets ×5 seeds |
| **P9** | Does DPO Help Classification? | β≤0.1: no effect, β=0.5: +11.8% | ✅ 4β×3 + 3λ×5 = 27 runs |
| **P14** | LoRA vs. OFT: A Task-Dependent Comparison | LoRA 0.91/OFT 0.86 | ✅ 20 exps |
| **P15** | Multi-Task Fine-Tuning as Implicit Regularization | 0.906±0.017 | ✅ 5 seeds |
| **P18** | Vocabulary Collapse in Fine-Tuned LLMs | 0.000 (20 folds, 5 seeds) | ✅ Domain Lock-In |
| **P19** | A 30-Item Reproducibility Checklist | — | ✅ Ready |
| **P20** | Cross-Domain Transferability of Fine-Tuned LLMs | AG 0.91/GoEmo 0.55 | ✅ |
| **P21** | Model Scale and Label Compliance | 0.910 | ✅ |
| **P22** | LoRA Rank, Data Quality, and Label Compliance | 0.881 | ✅ r=4→128 |
| **P23** | Post-Training Quantization Effects on Label Compliance | 0.909 (4=16bit) | ✅ No quant loss |
| **P24** | A Difficulty Grading Framework for Cybersecurity NLP Datasets | LLM value at H≈3-5 bits | ✅ |
| **P13** | SALAD-v2 (dataset paper) | — | ❌ Waiting P3 publish |

### P19 Self-Audit Results (14 papers audited)
- Mean score: **25.7/30** (85.7%)
- Most common failure: **Single seed** (Items 14–16)
- Multi-seed papers (P3,P6,P22,P15) score ≥25/30
- Automated tool `llm_eval_audit.py` tested ✅
- Figure generation: `generate_all_figures.py` produces all 42 PDFs

## Paper Finalization Workflow
> Run `/paper-finalize` on **every paper** before submission!
> 4 rounds: Proofread → Plagiarism check → Humanize → Verification
> See `.agents/workflows/paper-finalize.md`

## Workflow

### Kaggle (free, T4 GPU)
```bash
# Copy kaggle_q1_batch1.py into Kaggle notebook, run cells
# Results autosave to /kaggle/working/q1_results/summary.csv
```

### Vast.ai (paid, RTX 4090 / A100)
⚠️ **CRITICAL: ต้องใช้ standalone `.py` + supervisord — ห้ามใช้ bash heredoc + nohup**

```bash
# 1. เช่า GPU (4090 ~$0.30/h, A100 ~$1.50/h)
#    Image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel, Disk: 30GB

# 2. Upload data + script (.py ไม่ใช่ .sh)
scp -P <PORT> data/train_5k_clean.json data/test_held_out.json root@<HOST>:/workspace/salad_data/
scp -P <PORT> scripts/vast_p9.py root@<HOST>:/workspace/

# 3. Verify data (ต้องทำทุกครั้ง!)
ssh -p <PORT> root@<HOST> 'python3 -c "import json; d=json.load(open(\"/workspace/salad_data/train_5k_clean.json\")); print(f\"OK: {len(d)} samples\")"'

# 4. Setup supervisord (persistent — ไม่ตายตอน SSH disconnect)
ssh -p <PORT> root@<HOST> 'cat > /etc/supervisor/conf.d/job.conf << EOF
[program:job]
command=python3 /workspace/vast_p9.py
directory=/workspace
stdout_logfile=/workspace/job.log
stderr_logfile=/workspace/job.log
autostart=true
autorestart=false
startsecs=5
EOF
supervisorctl reread && supervisorctl update'

# 5. Check progress (ได้ตลอด ไม่ต้องค้าง SSH)
ssh -p <PORT> root@<HOST> 'supervisorctl status; tail -5 /workspace/job.log; nvidia-smi --query-gpu=memory.used --format=csv,noheader'

# 6. Download results + destroy instance
scp -P <PORT> -r root@<HOST>:/workspace/results/ results/vastai/
# แล้ว destroy ที่ https://cloud.vast.ai/instances/
```

### Vast.ai — Known Issues
| ปัญหา | สาเหตุ | แก้ไข |
|--------|--------|-------|
| `nohup` + SSH disconnect → process ตาย | Vast.ai kill child processes | ใช้ **supervisord** แทน |
| Bash heredoc `<< 'EOF'` ไม่ทำงาน | Shell escaping issues | เขียน **standalone .py** แทน |
| wget GitHub → empty file (0 bytes) | Rate limit / network | **SCP ตรงจาก local** + verify ทุกครั้ง |
| `auto-gptq` install ล้มเหลว | Build dependency missing | ลบออก, ใช้ bitsandbytes แทน |

### P19 Automated Audit
```bash
python3 scripts/llm_eval_audit.py --train train.jsonl --test test.jsonl --preds preds.jsonl --labels labels.txt
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
- **TRL 0.29 removed `ORPOTrainer` and `max_prompt_length`** — implement ORPO manually, use `max_length` only in DPOConfig
- **Qwen2.5-9B-Instruct is gated on HF** — needs HF token, or use 7B instead
- **P9 7B F1=1.0 (all 3 seeds)** — verified genuine: SALAD has only 35 unique test patterns, 7B memorizes all

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

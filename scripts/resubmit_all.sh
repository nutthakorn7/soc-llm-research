#!/bin/bash
# ============================================================
# Resubmit all FAILED/TIMEOUT Lanta jobs (March 12, 2026)
#
# FIXES APPLIED:
#   1. HF_HUB_OFFLINE=1 — forces transformers 5.2 to skip
#      HuggingFace hub validation on local model paths
#   2. Correct templates: SmolLM2=smollm, DeepSeek=deepseek
#   3. Uses clean_5k dataset (not salad_50k)
#   4. Proper checkpoint resume for 50K/20K
# ============================================================

PROJECT=/project/lt200473-ttctvs/soc-finetune
LLAMAFACTORY=/project/lt200473-ttctvs/workshop-pretrain/LlamaFactory
DATA=$PROJECT/data
MODELS=$PROJECT/models
OUT=$PROJECT/outputs
SCRIPTS=$PROJECT/scripts

# ============================================================
# 1. SmolLM2 multi-seed (s77, s999)
#    Template: smollm (verified from orchestrator.sh mm-smol-5k)
# ============================================================
for SEED in 77 999; do
cat > /tmp/fix-smol-s${SEED}.sh <<'OUTER'
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -A lt200473
OUTER
echo "#SBATCH -J smol-s${SEED}" >> /tmp/fix-smol-s${SEED}.sh
echo "#SBATCH -o $OUT/smol-s${SEED}_%j.out" >> /tmp/fix-smol-s${SEED}.sh
echo "#SBATCH -e $OUT/smol-s${SEED}_%j.err" >> /tmp/fix-smol-s${SEED}.sh
cat >> /tmp/fix-smol-s${SEED}.sh <<EOF

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $LLAMAFACTORY

llamafactory-cli train \\
    --model_name_or_path $MODELS/SmolLM2-1.7B-Instruct \\
    --quantization_bit 4 --quantization_method bnb \\
    --stage sft --do_train true \\
    --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \\
    --dataset_dir $DATA --dataset clean_5k --template smollm \\
    --cutoff_len 1024 \\
    --output_dir $OUT/seed-${SEED}-smol-5k \\
    --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \\
    --num_train_epochs 3 --learning_rate 2e-4 \\
    --warmup_ratio 0.1 --lr_scheduler_type cosine \\
    --seed $SEED \\
    --bf16 true --logging_steps 50 --save_steps 500 \\
    --plot_loss true --overwrite_output_dir true --report_to none
EOF
done

# ============================================================
# 2. DeepSeek multi-seed (s77, s999)
#    Template: deepseek (verified from orchestrator.sh mm-dsk-5k)
# ============================================================
for SEED in 77 999; do
cat > /tmp/fix-dsk-s${SEED}.sh <<'OUTER'
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -A lt200473
OUTER
echo "#SBATCH -J dsk-s${SEED}" >> /tmp/fix-dsk-s${SEED}.sh
echo "#SBATCH -o $OUT/dsk-s${SEED}_%j.out" >> /tmp/fix-dsk-s${SEED}.sh
echo "#SBATCH -e $OUT/dsk-s${SEED}_%j.err" >> /tmp/fix-dsk-s${SEED}.sh
cat >> /tmp/fix-dsk-s${SEED}.sh <<EOF

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $LLAMAFACTORY

llamafactory-cli train \\
    --model_name_or_path $MODELS/DeepSeek-R1-Distill-Qwen-7B \\
    --quantization_bit 4 --quantization_method bnb \\
    --stage sft --do_train true \\
    --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \\
    --dataset_dir $DATA --dataset clean_5k --template deepseek \\
    --cutoff_len 1024 \\
    --output_dir $OUT/seed-${SEED}-dsk-5k \\
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \\
    --num_train_epochs 3 --learning_rate 2e-4 \\
    --warmup_ratio 0.1 --lr_scheduler_type cosine \\
    --seed $SEED \\
    --bf16 true --logging_steps 50 --save_steps 500 \\
    --plot_loss true --overwrite_output_dir true --report_to none
EOF
done

# ============================================================
# 3. SmolLM2 LoRA rank ablation (r16, r32, r128)
# ============================================================
for RANK in 16 32 128; do
    ALPHA=$((RANK * 2))
cat > /tmp/fix-smol-r${RANK}.sh <<'OUTER'
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -A lt200473
OUTER
echo "#SBATCH -J smol-r${RANK}" >> /tmp/fix-smol-r${RANK}.sh
echo "#SBATCH -o $OUT/smol-r${RANK}_%j.out" >> /tmp/fix-smol-r${RANK}.sh
echo "#SBATCH -e $OUT/smol-r${RANK}_%j.err" >> /tmp/fix-smol-r${RANK}.sh
cat >> /tmp/fix-smol-r${RANK}.sh <<EOF

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $LLAMAFACTORY

llamafactory-cli train \\
    --model_name_or_path $MODELS/SmolLM2-1.7B-Instruct \\
    --quantization_bit 4 --quantization_method bnb \\
    --stage sft --do_train true \\
    --finetuning_type lora --lora_rank $RANK --lora_alpha $ALPHA --lora_target all \\
    --dataset_dir $DATA --dataset clean_5k --template smollm \\
    --cutoff_len 1024 \\
    --output_dir $OUT/abl-smol-r${RANK} \\
    --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \\
    --num_train_epochs 3 --learning_rate 2e-4 \\
    --warmup_ratio 0.1 --lr_scheduler_type cosine \\
    --bf16 true --logging_steps 50 --save_steps 500 \\
    --plot_loss true --overwrite_output_dir true --report_to none
EOF
done

# ============================================================
# 4. DeepSeek LoRA rank ablation (r16, r32, r128)
# ============================================================
for RANK in 16 32 128; do
    ALPHA=$((RANK * 2))
cat > /tmp/fix-dsk-r${RANK}.sh <<'OUTER'
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -A lt200473
OUTER
echo "#SBATCH -J dsk-r${RANK}" >> /tmp/fix-dsk-r${RANK}.sh
echo "#SBATCH -o $OUT/dsk-r${RANK}_%j.out" >> /tmp/fix-dsk-r${RANK}.sh
echo "#SBATCH -e $OUT/dsk-r${RANK}_%j.err" >> /tmp/fix-dsk-r${RANK}.sh
cat >> /tmp/fix-dsk-r${RANK}.sh <<EOF

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $LLAMAFACTORY

llamafactory-cli train \\
    --model_name_or_path $MODELS/DeepSeek-R1-Distill-Qwen-7B \\
    --quantization_bit 4 --quantization_method bnb \\
    --stage sft --do_train true \\
    --finetuning_type lora --lora_rank $RANK --lora_alpha $ALPHA --lora_target all \\
    --dataset_dir $DATA --dataset clean_5k --template deepseek \\
    --cutoff_len 1024 \\
    --output_dir $OUT/abl-dsk-r${RANK} \\
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \\
    --num_train_epochs 3 --learning_rate 2e-4 \\
    --warmup_ratio 0.1 --lr_scheduler_type cosine \\
    --bf16 true --logging_steps 50 --save_steps 500 \\
    --plot_loss true --overwrite_output_dir true --report_to none
EOF
done

# ============================================================
# 5. Resume 50K training (Qwen3.5-9B, from checkpoint-9375)
# ============================================================
cat > /tmp/fix-resume-50k.sh <<'EOF'
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 16:00:00
#SBATCH -A lt200473
#SBATCH -J resume-50k
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/resume-50k_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/resume-50k_%j.err

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /project/lt200473-ttctvs/workshop-pretrain/LlamaFactory

CKPT=$(ls -d /project/lt200473-ttctvs/soc-finetune/outputs/clean-qwen35-50k/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "Resuming from: $CKPT"

llamafactory-cli train \
    --model_name_or_path /project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-9B \
    --quantization_bit 4 --quantization_method bnb \
    --stage sft --do_train true \
    --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \
    --dataset_dir /project/lt200473-ttctvs/soc-finetune/data --dataset clean_50k --template qwen3_5 \
    --cutoff_len 1024 \
    --output_dir /project/lt200473-ttctvs/soc-finetune/outputs/clean-qwen35-50k \
    --resume_from_checkpoint $CKPT \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --num_train_epochs 3 --learning_rate 2e-4 \
    --warmup_ratio 0.1 --lr_scheduler_type cosine \
    --bf16 true --logging_steps 50 --save_steps 500 \
    --plot_loss true --report_to none
EOF

# ============================================================
# 6. Resume 20K training (Qwen3.5-9B, from checkpoint-3750)
# ============================================================
cat > /tmp/fix-resume-20k.sh <<'EOF'
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 08:00:00
#SBATCH -A lt200473
#SBATCH -J resume-20k
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/resume-20k_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/resume-20k_%j.err

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /project/lt200473-ttctvs/workshop-pretrain/LlamaFactory

CKPT=$(ls -d /project/lt200473-ttctvs/soc-finetune/outputs/clean-qwen35-20k/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "Resuming from: $CKPT"

llamafactory-cli train \
    --model_name_or_path /project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-9B \
    --quantization_bit 4 --quantization_method bnb \
    --stage sft --do_train true \
    --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \
    --dataset_dir /project/lt200473-ttctvs/soc-finetune/data --dataset clean_20k --template qwen3_5 \
    --cutoff_len 1024 \
    --output_dir /project/lt200473-ttctvs/soc-finetune/outputs/clean-qwen35-20k \
    --resume_from_checkpoint $CKPT \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --num_train_epochs 3 --learning_rate 2e-4 \
    --warmup_ratio 0.1 --lr_scheduler_type cosine \
    --bf16 true --logging_steps 50 --save_steps 500 \
    --plot_loss true --report_to none
EOF

echo ""
echo "============================================"
echo "  Scripts created in /tmp/fix-*.sh"
echo "============================================"
echo ""
echo "Files created:"
ls -la /tmp/fix-*.sh 2>/dev/null
echo ""
echo "--- UPLOAD & SUBMIT COMMANDS ---"
echo ""
echo "# Upload all scripts:"
echo 'for f in /tmp/fix-*.sh; do'
echo '  name=$(basename $f)'
echo '  ssh lanta "cat > /project/lt200473-ttctvs/soc-finetune/scripts/$name" < $f'
echo '  echo "  Uploaded $name"'
echo 'done'
echo ""
echo "# Submit all jobs:"
echo 'for f in /tmp/fix-*.sh; do'
echo '  name=$(basename $f)'
echo '  ssh lanta "cd /project/lt200473-ttctvs/soc-finetune && sbatch scripts/$name"'
echo 'done'

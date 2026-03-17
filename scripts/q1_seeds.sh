#!/bin/bash
# ===================================================================
# Q1 P0: เพิ่ม seeds สำหรับ P6, P15, P20, P21
# Submit: sbatch q1_seeds.sh
# Est. time: ~3h on 1×A100
# ===================================================================
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -A lt200473
#SBATCH -J q1-seeds
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/q1_seeds_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/q1_seeds_%j.err

echo "=== Q1 Seed Expansion ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(date)"

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
nvidia-smi

PROJECT=/project/lt200473-ttctvs/soc-finetune
SCRIPTS=$PROJECT/../soc-llm-research/scripts
cd /project/lt200473-ttctvs/workshop-pretrain/LlamaFactory

# =============================================
# P6: Qwen3.5-0.8B on SALAD — add seeds 456, 999 (has 42,77,123)
# =============================================
for SEED in 456 999; do
    echo ">>> P6: Qwen3.5-0.8B SALAD seed=$SEED $(date)"
    llamafactory-cli train \
        --model_name_or_path $PROJECT/models/Qwen3.5-0.8B \
        --quantization_bit 4 --quantization_method bnb \
        --stage sft --do_train true \
        --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \
        --dataset_dir $PROJECT/data \
        --dataset salad_clean_5k --template qwen3_5 \
        --cutoff_len 1024 \
        --output_dir $PROJECT/outputs/p6-qwen08b-seed$SEED \
        --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \
        --num_train_epochs 3 --learning_rate 2e-4 \
        --warmup_ratio 0.1 --lr_scheduler_type cosine \
        --seed $SEED \
        --bf16 true --logging_steps 50 --save_steps 9999 \
        --plot_loss true --overwrite_output_dir true --report_to none
    echo "<<< P6 seed=$SEED done $(date)"
done

# =============================================
# P15: Qwen3.5-0.8B Multi-Task + Single-Task — add seeds 456, 999 (has 42,77,123)
# =============================================
for SEED in 456 999; do
    echo ">>> P15: Multi-Task seed=$SEED $(date)"
    llamafactory-cli train \
        --model_name_or_path $PROJECT/models/Qwen3.5-0.8B \
        --quantization_bit 4 --quantization_method bnb \
        --stage sft --do_train true \
        --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \
        --dataset_dir $PROJECT/data \
        --dataset salad_multitask_5k --template qwen3_5 \
        --cutoff_len 1024 \
        --output_dir $PROJECT/outputs/p15-mt-seed$SEED \
        --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \
        --num_train_epochs 3 --learning_rate 2e-4 \
        --warmup_ratio 0.1 --lr_scheduler_type cosine \
        --seed $SEED \
        --bf16 true --logging_steps 50 --save_steps 9999 \
        --plot_loss true --overwrite_output_dir true --report_to none
    echo "<<< P15 MT seed=$SEED done $(date)"

    echo ">>> P15: Single-Task seed=$SEED $(date)"
    llamafactory-cli train \
        --model_name_or_path $PROJECT/models/Qwen3.5-0.8B \
        --quantization_bit 4 --quantization_method bnb \
        --stage sft --do_train true \
        --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \
        --dataset_dir $PROJECT/data \
        --dataset salad_clean_5k --template qwen3_5 \
        --cutoff_len 1024 \
        --output_dir $PROJECT/outputs/p15-st-seed$SEED \
        --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \
        --num_train_epochs 3 --learning_rate 2e-4 \
        --warmup_ratio 0.1 --lr_scheduler_type cosine \
        --seed $SEED \
        --bf16 true --logging_steps 50 --save_steps 9999 \
        --plot_loss true --overwrite_output_dir true --report_to none
    echo "<<< P15 ST seed=$SEED done $(date)"
done

# =============================================
# P20/P21: Qwen2.5-0.5B on AG News — add seed 999 (has 0,42,77,123)
# =============================================
echo ">>> P20/P21: Qwen2.5-0.5B AG News seed=999 $(date)"
python $SCRIPTS/train_crossdomain.py \
    --domain ag_news --seed 999 --train_size 5000 --test_size 1000 \
    --model $PROJECT/models/Qwen2.5-0.5B-Instruct \
    --epochs 3 --batch 4 --grad_acc 8 --rank 64
echo "<<< P20/P21 seed=999 done $(date)"

echo "=== ALL SEED JOBS COMPLETE ==="
echo "End: $(date)"

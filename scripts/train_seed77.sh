#!/bin/bash
# Train SmolLM2-1.7B with seeds 77 and 999 for 5-seed statistical rigor
# Creates 2 jobs: smol-seed77 and smol-seed999
# Usage: sbatch train_seeds.sh

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -A lt200473
#SBATCH -J smol-s77
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/seed77_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/seed77_%j.err

echo "=== SOC-FT: SmolLM2-1.7B Seed 77 ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(date)"

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
nvidia-smi

MODEL_DIR=/project/lt200473-ttctvs/soc-finetune/models/SmolLM2-1.7B-Instruct
cd /project/lt200473-ttctvs/workshop-pretrain/LlamaFactory

llamafactory-cli train \
    --model_name_or_path $MODEL_DIR \
    --quantization_bit 4 --quantization_method bnb \
    --stage sft --do_train true \
    --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \
    --dataset_dir /project/lt200473-ttctvs/soc-finetune/data \
    --dataset salad_clean_5k --template smollm2 \
    --cutoff_len 1024 \
    --output_dir /project/lt200473-ttctvs/soc-finetune/outputs/smol-seed77 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \
    --num_train_epochs 3 --learning_rate 2e-4 \
    --warmup_ratio 0.1 --lr_scheduler_type cosine \
    --seed 77 \
    --bf16 true --logging_steps 50 --save_steps 500 \
    --eval_strategy steps --eval_steps 500 --eval_dataset salad_val \
    --plot_loss true --overwrite_output_dir true --report_to none

echo "=== Seed 77 Complete === | End: $(date)"

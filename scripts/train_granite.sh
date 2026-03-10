#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 08:00:00
#SBATCH -A lt200473
#SBATCH -J soc-gran
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/train_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/train_%j.err

echo "=== SOC-FT: Granite-3.3-8B ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | GPUs: 4"
echo "Start: $(date)"

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
nvidia-smi

MODEL_DIR=/project/lt200473-ttctvs/soc-finetune/models/granite-3.3-8b-instruct

cd /project/lt200473-ttctvs/workshop-pretrain/LlamaFactory

llamafactory-cli train \
    --model_name_or_path $MODEL_DIR \
    --quantization_bit 4 --quantization_method bnb \
    --stage sft --do_train true \
    --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \
    --dataset_dir /project/lt200473-ttctvs/soc-finetune/data \
    --dataset salad_50k --template granite4 \
    --cutoff_len 1024 \
    --output_dir /project/lt200473-ttctvs/soc-finetune/outputs/granite-8b-lora \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --num_train_epochs 3 --learning_rate 2e-4 \
    --warmup_ratio 0.1 --lr_scheduler_type cosine \
    --bf16 true --logging_steps 50 --save_steps 500 \
    --eval_strategy steps --eval_steps 500 --eval_dataset salad_val \
    --plot_loss true --overwrite_output_dir true --report_to none

echo "=== Training Complete === | End: $(date)"

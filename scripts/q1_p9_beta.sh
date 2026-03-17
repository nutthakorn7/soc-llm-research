#!/bin/bash
# ===================================================================
# Q1 P0: P9 — DPO β sweep (0.001, 0.01, 0.05, 0.5)
# Submit: sbatch q1_p9_beta.sh
# Est. time: ~14h on 1×A100
# NOTE: DPO needs ~40GB VRAM (policy+reference), A100 80GB required
# ===================================================================
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -A lt200473
#SBATCH -J q1-p9-dpo
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/q1_p9_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/q1_p9_%j.err

echo "=== Q1 P9: DPO β Sweep ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(date)"

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
nvidia-smi

PROJECT=/project/lt200473-ttctvs/soc-finetune
cd /project/lt200473-ttctvs/workshop-pretrain/LlamaFactory

# Step 1: SFT baseline (if not already done)
SFT_MODEL=$PROJECT/outputs/p9-sft-base
if [ ! -d "$SFT_MODEL" ]; then
    echo ">>> Step 1: SFT baseline $(date)"
    llamafactory-cli train \
        --model_name_or_path $PROJECT/models/Qwen2.5-0.5B-Instruct \
        --quantization_bit 4 --quantization_method bnb \
        --stage sft --do_train true \
        --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \
        --dataset_dir $PROJECT/data \
        --dataset salad_clean_5k --template qwen \
        --cutoff_len 1024 \
        --output_dir $SFT_MODEL \
        --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \
        --num_train_epochs 3 --learning_rate 2e-4 \
        --warmup_ratio 0.1 --lr_scheduler_type cosine \
        --seed 42 \
        --bf16 true --logging_steps 50 --save_steps 9999 \
        --plot_loss true --overwrite_output_dir true --report_to none
    echo "<<< SFT baseline done $(date)"
fi

# Step 2: Generate preference pairs from SFT errors
echo ">>> Step 2: Generate preference pairs $(date)"
# TODO: Run eval on SFT model, extract hallucination pairs
# This should produce a JSONL file at $PROJECT/data/dpo_prefs.jsonl
# Format: {"prompt": "...", "chosen": "Reconnaissance", "rejected": "Port Scanning"}

# Step 3: DPO with different β values
for BETA in 0.001 0.01 0.05 0.5; do
    for SEED in 42 77 123; do
        echo ">>> DPO β=$BETA seed=$SEED $(date)"
        llamafactory-cli train \
            --model_name_or_path $PROJECT/models/Qwen2.5-0.5B-Instruct \
            --adapter_name_or_path $SFT_MODEL \
            --quantization_bit 4 --quantization_method bnb \
            --stage dpo --do_train true \
            --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \
            --dataset_dir $PROJECT/data \
            --dataset dpo_prefs --template qwen \
            --cutoff_len 1024 \
            --output_dir $PROJECT/outputs/p9-dpo-beta${BETA}-seed${SEED} \
            --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
            --num_train_epochs 1 --learning_rate 5e-5 \
            --dpo_beta $BETA \
            --seed $SEED \
            --bf16 true --logging_steps 10 --save_steps 9999 \
            --plot_loss true --overwrite_output_dir true --report_to none
        echo "<<< DPO β=$BETA seed=$SEED done $(date)"
    done
done

echo "=== P9 β SWEEP COMPLETE ==="
echo "End: $(date)"

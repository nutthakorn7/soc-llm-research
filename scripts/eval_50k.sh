#!/bin/bash
# eval_50k.sh — Evaluate the 50K model immediately after training completes
# Usage: sbatch eval_50k.sh
# Or run manually: bash eval_50k.sh

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -A lt200473
#SBATCH -J ev-50k
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/eval50k_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/eval50k_%j.err

echo "=== Eval: Qwen3.5-9B 50K ==="
echo "Job ID: $SLURM_JOB_ID | $(date)"

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune

BASE=/project/lt200473-ttctvs/soc-finetune
MODEL=$BASE/models/Qwen3.5-0.8B  
ADAPTER=$BASE/outputs/clean-qwen35-50k
TEMPLATE=qwen3_5
OUTNAME=eval-50k-strict

# Check adapter exists
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json not found. Training may not be complete."
    echo "Checking for checkpoint..."
    LATEST_CKPT=$(ls -td $ADAPTER/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Using latest checkpoint: $LATEST_CKPT"
        ADAPTER=$LATEST_CKPT
    else
        echo "FATAL: No adapter or checkpoint found."
        exit 1
    fi
fi

cd /project/lt200473-ttctvs/workshop-pretrain/LlamaFactory

echo "=== Running inference ==="
llamafactory-cli train \
    --model_name_or_path $MODEL \
    --adapter_name_or_path $ADAPTER \
    --template $TEMPLATE \
    --quantization_bit 4 --quantization_method bnb \
    --stage sft --do_predict true \
    --dataset_dir $BASE/data \
    --dataset salad_test --cutoff_len 1024 \
    --output_dir $BASE/outputs/$OUTNAME \
    --per_device_eval_batch_size 8 \
    --bf16 true --predict_with_generate true \
    --max_new_tokens 128 --report_to none

echo "=== Inference complete ==="

# Run scoring
echo "=== Computing strict vs normalized F1 ==="
python3 $BASE/scripts/compute_f1.py \
    --predictions $BASE/outputs/$OUTNAME/generated_predictions.jsonl \
    --output $BASE/outputs/$OUTNAME/scores.json 2>/dev/null || \
    echo "compute_f1.py not found — run manually"

echo "=== Done: $(date) ==="

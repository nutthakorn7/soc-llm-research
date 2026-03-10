#!/bin/bash
# fast_eval.sh — Faster eval using LlamaFactory chat + batching
# Usage: sbatch fast_eval.sh <model_path> <adapter_dir> <template> <eval_name>
# Example: sbatch fast_eval.sh models/Qwen3.5-9B outputs/clean-qwen35-5k qwen3_5 clean-qwen35-5k
#
# Key difference from eval.sh:
#   - Uses predict with max_new_tokens=150 (SOC response is ~80 tokens, not 300)
#   - Adds per_device_eval_batch_size=8 for batch inference
#   - ~2-3x faster than default eval.sh

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -A lt200473
#SBATCH -J soc-fast-eval
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/fast_eval_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/fast_eval_%j.err

MODEL=$1      # e.g., models/Qwen3.5-9B
ADAPTER=$2    # e.g., outputs/clean-qwen35-5k
TEMPLATE=$3   # e.g., qwen3_5
EVAL_NAME=$4  # e.g., clean-qwen35-5k

BASE=/project/lt200473-ttctvs/soc-finetune
LLAMA=/project/lt200473-ttctvs/workshop-pretrain/LlamaFactory
OUTPUT=$BASE/outputs/eval-$EVAL_NAME

echo "=== Fast Eval ==="
echo "Model: $MODEL"
echo "Adapter: $ADAPTER"
echo "Template: $TEMPLATE"
echo "Output: $OUTPUT"
echo "Start: $(date)"
echo ""

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune

cd $LLAMA

# Key optimizations:
# 1. max_new_tokens=150 (not 512) — SOC triage response is ~80 tokens
# 2. per_device_eval_batch_size=8 — process 8 samples at once
# 3. predict_with_generate uses greedy decoding (faster)
llamafactory-cli train \
  --stage sft \
  --model_name_or_path $BASE/$MODEL \
  --adapter_name_or_path $BASE/$ADAPTER \
  --template $TEMPLATE \
  --dataset_dir $BASE/data \
  --dataset clean_test \
  --output_dir $OUTPUT \
  --do_predict \
  --per_device_eval_batch_size 8 \
  --max_new_tokens 150 \
  --quantization_method bnb \
  --quantization_bit 4 \
  --predict_with_generate \
  --finetuning_type lora \
  --overwrite_output_dir

echo ""
echo "=== Done ==="
echo "Predictions: $OUTPUT/generated_predictions.jsonl"
echo "End: $(date)"

# Auto calc F1 if predictions exist
if [ -f "$OUTPUT/generated_predictions.jsonl" ]; then
    lines=$(wc -l < "$OUTPUT/generated_predictions.jsonl")
    echo "Predictions: $lines lines"
    
    if [ -f "$BASE/scripts/calc_f1.py" ]; then
        echo "Running F1 calc..."
        python3 $BASE/scripts/calc_f1.py $BASE/data/test_held_out.json $OUTPUT/generated_predictions.jsonl > $OUTPUT/f1_results.json 2>&1
        echo "F1 results: $OUTPUT/f1_results.json"
        cat $OUTPUT/f1_results.json | head -20
    fi
fi

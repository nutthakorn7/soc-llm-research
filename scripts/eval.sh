#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -A lt200473
#SBATCH -J soc-eval
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/eval_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/eval_%j.err

# Usage: sbatch eval.sh <model_dir> <adapter_dir> <template> <output_name>
MODEL_DIR=${1:-models/Qwen3.5-9B}
ADAPTER_DIR=${2:-outputs/scale-qwen35-5k}
TEMPLATE=${3:-qwen3_5}
OUTPUT_NAME=${4:-eval_result}

PROJECT=/project/lt200473-ttctvs/soc-finetune

echo "=== SOC-FT Evaluation ==="
echo "Model: $MODEL_DIR"
echo "Adapter: $ADAPTER_DIR"
echo "Template: $TEMPLATE"
echo "Output: $OUTPUT_NAME"
echo "Start: $(date)"

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune

cd /project/lt200473-ttctvs/workshop-pretrain/LlamaFactory

# Use LlamaFactory predict stage
llamafactory-cli train \
    --model_name_or_path ${PROJECT}/${MODEL_DIR} \
    --adapter_name_or_path ${PROJECT}/${ADAPTER_DIR} \
    --quantization_bit 4 --quantization_method bnb \
    --stage sft --do_predict true --do_train false \
    --finetuning_type lora \
    --dataset_dir ${PROJECT}/data \
    --dataset salad_val \
    --eval_dataset salad_val \
    --template $TEMPLATE \
    --cutoff_len 1024 \
    --output_dir ${PROJECT}/outputs/eval-${OUTPUT_NAME} \
    --per_device_eval_batch_size 4 \
    --predict_with_generate true \
    --max_new_tokens 128 \
    --bf16 true \
    --report_to none \
    --overwrite_output_dir true

echo "=== Prediction files ==="
ls -la ${PROJECT}/outputs/eval-${OUTPUT_NAME}/

# Check what files were generated
echo "=== Generated files ==="
find ${PROJECT}/outputs/eval-${OUTPUT_NAME}/ -type f -name "*.json*" | while read f; do
    echo "$f: $(wc -l < $f) lines"
    head -2 $f
    echo "..."
done

echo "=== Evaluation Complete === | End: $(date)"

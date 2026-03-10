#!/bin/bash
#SBATCH -p gpu-devel
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -A lt200473
#SBATCH -J soc-smoke
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/smoke_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/smoke_%j.err

echo "=== Smoke Test: SOC Alert Fine-tuning ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

# Load modules
module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune

nvidia-smi

# === Use LOCAL pre-downloaded model (GPU nodes have no internet) ===
MODEL_DIR=/project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-9B

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "Download first on login node!"
    exit 1
fi

echo "Using local model: $MODEL_DIR"

# Create a tiny training set (100 samples) from the 50K set
python3 -c "
import json
with open('/project/lt200473-ttctvs/soc-finetune/data/train_50k.json') as f:
    data = json.load(f)
tiny = data[:100]
with open('/project/lt200473-ttctvs/soc-finetune/data/train_tiny.json', 'w') as f:
    json.dump(tiny, f)
print(f'Created tiny dataset: {len(tiny)} samples')
"

# Create tiny dataset_info entry
python3 -c "
import json
info_path = '/project/lt200473-ttctvs/soc-finetune/data/dataset_info.json'
with open(info_path) as f:
    info = json.load(f)
info['salad_tiny'] = dict(info['salad_50k'])
info['salad_tiny']['file_name'] = '/project/lt200473-ttctvs/soc-finetune/data/train_tiny.json'
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)
print('Added salad_tiny to dataset_info.json')
"

# Run training with tiny config
LLAMAFACTORY_DIR=/project/lt200473-ttctvs/workshop-pretrain/LlamaFactory
cd $LLAMAFACTORY_DIR

llamafactory-cli train \
    --model_name_or_path $MODEL_DIR \
    --quantization_bit 4 \
    --quantization_method bnb \
    --stage sft \
    --do_train true \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_target all \
    --dataset_dir /project/lt200473-ttctvs/soc-finetune/data \
    --dataset salad_tiny \
    --template qwen3_5 \
    --cutoff_len 1024 \
    --output_dir /project/lt200473-ttctvs/soc-finetune/outputs/smoke-test \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --bf16 true \
    --logging_steps 10 \
    --save_steps 100 \
    --overwrite_output_dir true \
    --report_to none

echo "=== Smoke Test Complete ==="
echo "End: $(date)"

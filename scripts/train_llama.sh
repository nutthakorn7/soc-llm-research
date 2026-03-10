#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 08:00:00
#SBATCH -A lt200473
#SBATCH -J soc-ft-llm
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/train_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/train_%j.err

echo "=== SOC-FT: Llama-3.1-8B Full Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "GPUs per node: 4"
echo "Total GPUs: 16"
echo "Start: $(date)"

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune

nvidia-smi

MODEL_DIR=/project/lt200473-ttctvs/soc-finetune/models/Llama-3.1-8B-Instruct

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=16
export NCCL_DEBUG=INFO

LLAMAFACTORY_DIR=/project/lt200473-ttctvs/workshop-pretrain/LlamaFactory
cd $LLAMAFACTORY_DIR

srun torchrun \
    --nproc_per_node=4 \
    --nnodes=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m llamafactory.cli train \
    --model_name_or_path $MODEL_DIR \
    --quantization_bit 4 \
    --quantization_method bnb \
    --stage sft \
    --do_train true \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_target all \
    --dataset_dir /project/lt200473-ttctvs/soc-finetune/data \
    --dataset salad_50k \
    --template llama3 \
    --cutoff_len 1024 \
    --output_dir /project/lt200473-ttctvs/soc-finetune/outputs/llama31-8b-lora \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_strategy steps \
    --eval_steps 500 \
    --eval_dataset salad_val \
    --plot_loss true \
    --overwrite_output_dir true \
    --report_to none

echo "=== Training Complete ==="
echo "End: $(date)"

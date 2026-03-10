#!/bin/bash
# === Scaling Law Experiments for Qwen3.5-9B ===
# Submit multiple jobs with different data sizes: 1K, 5K, 10K, 20K
# (50K already submitted separately)

PROJECT=/project/lt200473-ttctvs/soc-finetune
MODEL_DIR=$PROJECT/models/Qwen3.5-9B
LLAMAFACTORY=/project/lt200473-ttctvs/workshop-pretrain/LlamaFactory

for SIZE in 1k 5k 10k 20k; do
    cat > /tmp/train_scale_${SIZE}.sh << EOF
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 08:00:00
#SBATCH -A lt200473
#SBATCH -J sc-${SIZE}
#SBATCH -o ${PROJECT}/outputs/scale_${SIZE}_%j.out
#SBATCH -e ${PROJECT}/outputs/scale_${SIZE}_%j.err

echo "=== Scaling Law: Qwen3.5-9B × ${SIZE} ==="
echo "Job ID: \$SLURM_JOB_ID | Node: \$SLURM_NODELIST"
echo "Start: \$(date)"

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune

cd ${LLAMAFACTORY}

llamafactory-cli train \\
    --model_name_or_path ${MODEL_DIR} \\
    --quantization_bit 4 --quantization_method bnb \\
    --stage sft --do_train true \\
    --finetuning_type lora --lora_rank 64 --lora_alpha 128 --lora_target all \\
    --dataset_dir ${PROJECT}/data \\
    --dataset salad_${SIZE} --template qwen3_5 \\
    --cutoff_len 1024 \\
    --output_dir ${PROJECT}/outputs/scale-qwen35-${SIZE} \\
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \\
    --num_train_epochs 3 --learning_rate 2e-4 \\
    --warmup_ratio 0.1 --lr_scheduler_type cosine \\
    --bf16 true --logging_steps 50 --save_steps 500 \\
    --eval_strategy steps --eval_steps 500 --eval_dataset salad_val \\
    --plot_loss true --overwrite_output_dir true --report_to none

echo "=== Done === | End: \$(date)"
EOF
    echo "Created train_scale_${SIZE}.sh"
done

echo "=== All scaling scripts created ==="

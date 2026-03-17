#!/bin/bash
# ===================================================================
# Q1 P0: P8 — Fill TBD results (LLM on AG News, GoEmotions, LedGAR)
# Submit: sbatch q1_p8_tbd.sh
# Est. time: ~6.5h on 1×A100
# ===================================================================
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -A lt200473
#SBATCH -J q1-p8
#SBATCH -o /project/lt200473-ttctvs/soc-finetune/outputs/q1_p8_%j.out
#SBATCH -e /project/lt200473-ttctvs/soc-finetune/outputs/q1_p8_%j.err

echo "=== Q1 P8: LLM on 3 Datasets (Fill TBD) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(date)"

module load cuda/12.6 Mamba/23.11.0-0
source activate soc-finetune
nvidia-smi

PROJECT=/project/lt200473-ttctvs/soc-finetune
SCRIPTS=$PROJECT/../soc-llm-research/scripts

# Qwen3.5-0.8B on 3 datasets × 5 seeds
# AG News: H(Y)=2.0 bits, 4 classes
# GoEmotions: H(Y)=3.75 bits, 28 classes
# LedGAR: H(Y)=6.16 bits, 100 classes

for DOMAIN in ag_news go_emotions ledgar; do
    for SEED in 42 77 123 456 999; do
        echo ">>> P8: $DOMAIN seed=$SEED $(date)"
        python $SCRIPTS/train_crossdomain.py \
            --domain $DOMAIN --seed $SEED \
            --train_size 5000 --test_size 1000 \
            --model $PROJECT/models/Qwen3.5-0.8B \
            --epochs 3 --batch 4 --grad_acc 8 --rank 64
        echo "<<< $DOMAIN seed=$SEED done $(date)"
    done
done

echo "=== P8 ALL DATASETS COMPLETE ==="
echo "End: $(date)"

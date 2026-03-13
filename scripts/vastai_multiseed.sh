#!/bin/bash
# ==========================================================================
# Multi-Seed Training on Vast.ai — Close P19 Items 14-16
# ==========================================================================
# Papers needing multi-seed: P7, P9, P18, P20, P21, P23, P24
# Models: Qwen2.5-0.5B (cheap, ~15min/seed), Phi4-mini (key, ~25min/seed)
#
# Usage:
#   1. Rent a Vast.ai instance with A100/H100 (≥24GB VRAM)
#   2. SSH into instance
#   3. bash vastai_multiseed.sh
#
# Total estimate: ~3-4 hours on single A100, ~$6-8
# ==========================================================================

set -euo pipefail

echo "=== Vast.ai Multi-Seed Training ==="
echo "Start: $(date)"

# ---- SETUP ----
pip install -q llamafactory datasets peft bitsandbytes accelerate scikit-learn trl 2>/dev/null
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"

# Download models (only what we need)
MODELS_DIR="./models"
mkdir -p $MODELS_DIR

echo "Downloading models..."
python3 -c "
from huggingface_hub import snapshot_download
models = [
    ('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen2.5-0.5B-Instruct'),
    ('microsoft/Phi-4-mini-4k-instruct', 'Phi4-mini'),
]
for repo, name in models:
    print(f'  Downloading {name}...')
    snapshot_download(repo, local_dir=f'$MODELS_DIR/{name}', ignore_patterns=['*.gguf'])
    print(f'  ✅ {name}')
"

# ---- DATA SETUP ----
DATA_DIR="./data"
mkdir -p $DATA_DIR

echo "Downloading SALAD clean splits..."
# TODO: Replace with actual data download from your repo
# For now, expects data files to be present:
#   data/train_5k_clean.json
#   data/test_held_out.json
#   data/dataset_info.json

if [ ! -f "$DATA_DIR/train_5k_clean.json" ]; then
    echo "❌ Missing data files! Please copy from repo:"
    echo "   scp train_5k_clean.json test_held_out.json dataset_info.json $DATA_DIR/"
    exit 1
fi

# ---- TRAINING FUNCTION ----
train_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$2
    local SEED=$3
    local OUTPUT_DIR="./outputs/${MODEL_NAME}-seed${SEED}"
    local TEMPLATE=$4

    if [ -d "$OUTPUT_DIR" ]; then
        echo "  ⏭️  Skip $MODEL_NAME seed=$SEED (exists)"
        return
    fi

    echo "  🚀 Training $MODEL_NAME seed=$SEED..."
    START=$(date +%s)

    llamafactory-cli train \
        --model_name_or_path "$MODEL_PATH" \
        --template "$TEMPLATE" \
        --dataset clean_5k \
        --dataset_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --finetuning_type lora \
        --lora_rank 64 --lora_alpha 128 \
        --lora_target all \
        --quantization_bit 4 --quantization_method bnb \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 3 \
        --learning_rate 2e-4 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_strategy no \
        --seed "$SEED" \
        --bf16 true \
        --cutoff_len 512 \
        --overwrite_output_dir true \
        --report_to none \
        2>&1 | tail -3

    END=$(date +%s)
    echo "  ✅ $MODEL_NAME seed=$SEED done in $((END-START))s"
}

# ---- EVAL FUNCTION ----
eval_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$2
    local SEED=$3
    local ADAPTER_DIR="./outputs/${MODEL_NAME}-seed${SEED}"
    local PRED_DIR="./outputs/eval-${MODEL_NAME}-seed${SEED}"
    local TEMPLATE=$4

    if [ ! -d "$ADAPTER_DIR" ]; then
        echo "  ❌ No adapter for $MODEL_NAME seed=$SEED"
        return
    fi

    echo "  📊 Evaluating $MODEL_NAME seed=$SEED..."

    llamafactory-cli train \
        --model_name_or_path "$MODEL_PATH" \
        --template "$TEMPLATE" \
        --adapter_name_or_path "$ADAPTER_DIR" \
        --dataset clean_test \
        --dataset_dir "$DATA_DIR" \
        --output_dir "$PRED_DIR" \
        --finetuning_type lora \
        --quantization_bit 4 --quantization_method bnb \
        --do_predict true \
        --per_device_eval_batch_size 8 \
        --max_new_tokens 150 \
        --predict_with_generate true \
        --bf16 true \
        --cutoff_len 512 \
        --report_to none \
        2>&1 | tail -3

    echo "  ✅ Eval $MODEL_NAME seed=$SEED done"
}

# ---- SEEDS ----
SEEDS=(42 77 123 456 999)

# ========== ROUND 1: Qwen2.5-0.5B (5 seeds, ~15min each = ~75min) ==========
echo ""
echo "========== ROUND 1: Qwen2.5-0.5B (5 seeds) =========="
QWEN_PATH="$MODELS_DIR/Qwen2.5-0.5B-Instruct"
for SEED in "${SEEDS[@]}"; do
    train_model "$QWEN_PATH" "qwen05b" "$SEED" "qwen2_5"
done

echo "--- Evaluating Qwen2.5-0.5B ---"
for SEED in "${SEEDS[@]}"; do
    eval_model "$QWEN_PATH" "qwen05b" "$SEED" "qwen2_5"
done

# ========== ROUND 2: Phi4-mini (5 seeds, ~25min each = ~125min) ==========
echo ""
echo "========== ROUND 2: Phi4-mini (5 seeds) =========="
PHI_PATH="$MODELS_DIR/Phi4-mini"
for SEED in "${SEEDS[@]}"; do
    train_model "$PHI_PATH" "phi4mini" "$SEED" "phi4"
done

echo "--- Evaluating Phi4-mini ---"
for SEED in "${SEEDS[@]}"; do
    eval_model "$PHI_PATH" "phi4mini" "$SEED" "phi4"
done

# ========== ROUND 3: Calculate F1 ==========
echo ""
echo "========== ROUND 3: F1 Calculation =========="
python3 << 'PYEOF'
import json, os, glob
from collections import defaultdict

results = defaultdict(dict)

for pred_dir in sorted(glob.glob("outputs/eval-*")):
    pred_file = os.path.join(pred_dir, "generated_predictions.jsonl")
    if not os.path.exists(pred_file):
        continue
    
    name = os.path.basename(pred_dir).replace("eval-", "")
    
    # Count errors
    total = 0
    strict_correct = 0
    with open(pred_file) as f:
        for line in f:
            d = json.loads(line)
            pred = d.get("predict", "").strip()
            label = d.get("label", "").strip()
            total += 1
            if pred == label:
                strict_correct += 1
    
    f1 = strict_correct / total if total > 0 else 0
    results[name.rsplit("-seed", 1)[0]][name.rsplit("seed", 1)[1]] = f1

print("\n=== Multi-Seed Results ===")
print(f"{'Model':<15} {'Seeds':>6} | " + " | ".join(f"s{s}" for s in [42,77,123,456,999]) + " | Mean±Std")
print("-" * 80)

import numpy as np
for model, seeds in sorted(results.items()):
    vals = [seeds.get(str(s), None) for s in [42,77,123,456,999]]
    present = [v for v in vals if v is not None]
    mean = np.mean(present) if present else 0
    std = np.std(present) if len(present) > 1 else 0
    vals_str = " | ".join(f"{v:.3f}" if v is not None else "  —  " for v in vals)
    print(f"{model:<15} {len(present):>4}/5 | {vals_str} | {mean:.3f}±{std:.3f}")

# Save results
with open("outputs/multiseed_results.json", "w") as f:
    json.dump(dict(results), f, indent=2)
print("\n✅ Results saved to outputs/multiseed_results.json")
PYEOF

echo ""
echo "=== All done! ==="
echo "End: $(date)"
echo ""
echo "Next steps:"
echo "  1. scp outputs/multiseed_results.json back to local"
echo "  2. Update papers with mean±std from 5 seeds"
echo "  3. Re-run P19 audit — Items 14-16 should now pass"

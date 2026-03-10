#!/bin/bash
# Auto-submit eval jobs when training completes
# Usage: bash auto_eval_chain.sh

PROJECT=/project/lt200473-ttctvs/soc-finetune
echo "=== Auto-Eval Chain $(date) ==="

submit_eval() {
    local adapter=$1 model=$2 template=$3
    local adapter_dir="$PROJECT/outputs/$adapter"
    local eval_dir="$PROJECT/outputs/eval-$adapter"
    
    if [ -f "$adapter_dir/adapter_config.json" ] && [ ! -d "$eval_dir" ]; then
        echo "  SUBMIT eval for $adapter"
        sbatch $PROJECT/scripts/eval.sh $model $adapter $template $adapter
    elif [ -d "$eval_dir" ] && [ -f "$eval_dir/generated_predictions.jsonl" ]; then
        lines=$(wc -l < "$eval_dir/generated_predictions.jsonl")
        echo "  DONE $adapter ($lines predictions)"
    elif [ -d "$eval_dir" ]; then
        echo "  RUNNING $adapter"
    elif [ -f "$adapter_dir/adapter_config.json" ]; then
        echo "  READY $adapter (eval exists)"
    else
        echo "  WAIT $adapter"
    fi
}

# Scaling law
submit_eval clean-qwen35-1k models/Qwen3.5-9B qwen3_5
submit_eval clean-qwen35-5k models/Qwen3.5-9B qwen3_5
submit_eval clean-qwen35-10k models/Qwen3.5-9B qwen3_5
submit_eval clean-qwen35-20k models/Qwen3.5-9B qwen3_5

# Multi-seed
submit_eval seed-123-q35-5k models/Qwen3.5-9B qwen3_5
submit_eval seed-2024-q35-5k models/Qwen3.5-9B qwen3_5

# Multi-model
submit_eval mm-dsk-5k models/DeepSeek-R1-Distill-Qwen-7B deepseek
submit_eval mm-phi4-5k models/Phi-4-mini-instruct phi4
submit_eval mm-gra-5k models/granite-3.3-8b-instruct granite
submit_eval mm-smol-5k models/SmolLM2-1.7B-Instruct smollm
submit_eval mm-gem-5k models/gemma-3-4b-it gemma3
submit_eval mm-mis-5k models/Mistral-7B-Instruct-v0.3 mistral
submit_eval mm-q3-5k models/Qwen3-8B qwen3
submit_eval mm-q08-5k models/Qwen3.5-0.8B qwen3_5

# Ablation
submit_eval abl-rank16 models/Qwen3.5-9B qwen3_5
submit_eval abl-rank32 models/Qwen3.5-9B qwen3_5
submit_eval abl-rank128 models/Qwen3.5-9B qwen3_5
submit_eval abl-lr1e-4 models/Qwen3.5-9B qwen3_5
submit_eval abl-lr5e-4 models/Qwen3.5-9B qwen3_5

echo ""
echo "=== Jobs ==="
squeue -u lm2002 -o "%.10i %.12j %.8T %.10M"

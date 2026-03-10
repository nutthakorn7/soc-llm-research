#!/bin/bash
# SOC-FT Orchestrator — monitors, auto-evals, auto-F1
# Usage: ssh lanta 'nohup bash /project/.../scripts/orchestrator.sh > /project/.../outputs/orchestrator.log 2>&1 &'
# Or:   ssh lanta 'bash /project/.../scripts/orchestrator.sh'

PROJECT=/project/lt200473-ttctvs/soc-finetune
EVAL_SCRIPT=$PROJECT/scripts/eval.sh
INTERVAL=120  # Check every 2 minutes

# Model configs: adapter_name|model_path|template
EXPERIMENTS=(
    "clean-qwen35-1k|models/Qwen3.5-9B|qwen3_5"
    "clean-qwen35-5k|models/Qwen3.5-9B|qwen3_5"
    "clean-qwen35-10k|models/Qwen3.5-9B|qwen3_5"
    "clean-qwen35-20k|models/Qwen3.5-9B|qwen3_5"
    "seed-123-q35-5k|models/Qwen3.5-9B|qwen3_5"
    "seed-2024-q35-5k|models/Qwen3.5-9B|qwen3_5"
    "mm-dsk-5k|models/DeepSeek-R1-Distill-Qwen-7B|deepseek"
    "mm-phi4-5k|models/Phi-4-mini-instruct|phi4"
    "mm-gra-5k|models/granite-3.3-8b-instruct|granite"
    "mm-smol-5k|models/SmolLM2-1.7B-Instruct|smollm"
    "mm-gem-5k|models/gemma-3-4b-it|gemma3"
    "mm-mis-5k|models/Mistral-7B-Instruct-v0.3|mistral"
    "mm-q3-5k|models/Qwen3-8B|qwen3"
    "mm-q08-5k|models/Qwen3.5-0.8B|qwen3_5"
    "abl-rank16|models/Qwen3.5-9B|qwen3_5"
    "abl-rank32|models/Qwen3.5-9B|qwen3_5"
    "abl-rank128|models/Qwen3.5-9B|qwen3_5"
    "abl-lr1e-4|models/Qwen3.5-9B|qwen3_5"
    "abl-lr5e-4|models/Qwen3.5-9B|qwen3_5"
)

log() { echo "[$(date '+%H:%M:%S')] $1"; }

check_and_act() {
    local trained=0 evaling=0 done=0 waiting=0 total=${#EXPERIMENTS[@]}
    
    log "--- Status Check ---"
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r adapter model template <<< "$exp"
        local adapter_dir="$PROJECT/outputs/$adapter"
        local eval_dir="$PROJECT/outputs/eval-$adapter"
        local preds="$eval_dir/generated_predictions.jsonl"
        local f1_file="$eval_dir/f1_results.json"
        
        # Stage 1: Training done? → Submit eval
        if [ -f "$adapter_dir/adapter_config.json" ] && [ ! -d "$eval_dir" ]; then
            log "  📤 $adapter: TRAINED → submitting eval"
            sbatch $EVAL_SCRIPT $model "outputs/$adapter" $template $adapter
            ((trained++))
        # Stage 2: Eval done? → Run F1
        elif [ -f "$preds" ] && [ ! -f "$f1_file" ]; then
            lines=$(wc -l < "$preds")
            log "  📊 $adapter: EVAL DONE ($lines preds) → running F1"
            module load Mamba/23.11.0-0 2>/dev/null
            source activate soc-finetune 2>/dev/null
            python3 $PROJECT/scripts/calc_f1.py $PROJECT/data/test_held_out.json $preds > $f1_file 2>&1
            log "  ✅ $adapter: F1 calculated"
            ((done++))
        # Stage 3: F1 done
        elif [ -f "$f1_file" ]; then
            ((done++))
        # Stage 4: Eval running
        elif [ -d "$eval_dir" ]; then
            ((evaling++))
        # Stage 5: Still waiting
        else
            ((waiting++))
        fi
    done
    
    log "  Summary: done=$done eval=$evaling trained=$trained wait=$waiting / $total"
    
    # Check if ALL done
    if [ $done -eq $total ]; then
        log "🎉 ALL EXPERIMENTS COMPLETE!"
        log "Running master_eval.py..."
        python3 $PROJECT/scripts/master_eval.py $PROJECT
        return 1  # Signal to stop
    fi
    return 0
}

# Main loop
log "🚀 SOC-FT Orchestrator started"
log "  Monitoring ${#EXPERIMENTS[@]} experiments every ${INTERVAL}s"
log ""

while true; do
    check_and_act
    status=$?
    if [ $status -eq 1 ]; then
        log "Orchestrator finished."
        break
    fi
    
    # Show running jobs
    running=$(squeue -u lm2002 -h | wc -l)
    log "  Active Slurm jobs: $running"
    
    if [ $running -eq 0 ]; then
        log "⚠️  No jobs running — checking if everything completed..."
        check_and_act
        if [ $? -eq 0 ]; then
            log "  Some experiments still waiting — may need manual resubmit"
        fi
    fi
    
    log "  Sleeping ${INTERVAL}s..."
    sleep $INTERVAL
done

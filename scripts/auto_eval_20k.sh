#!/bin/bash
# auto_eval_20k.sh — Monitor 20K training and auto-submit eval when done
# Usage: Run on local Mac: bash scripts/auto_eval_20k.sh
# It polls Lanta every 2 minutes and submits eval when training finishes.

TRAIN_JOB=4797582
REMOTE="lanta"
BASE="/project/lt200473-ttctvs/soc-finetune"

echo "======================================================================="
echo "  Monitoring 20K training (Job $TRAIN_JOB)"
echo "  Will auto-submit eval when training completes"
echo "======================================================================="

while true; do
    # Check if training job is still running
    STATUS=$(ssh $REMOTE "squeue -j $TRAIN_JOB --noheader -o '%T' 2>/dev/null" | tr -d '[:space:]')
    
    if [ -z "$STATUS" ] || [ "$STATUS" = "" ]; then
        echo ""
        echo "🎉 Training job $TRAIN_JOB completed!"
        
        # Verify adapter exists
        ADAPTER=$(ssh $REMOTE "ls $BASE/outputs/clean-qwen35-20k/adapter_config.json 2>/dev/null")
        if [ -z "$ADAPTER" ]; then
            echo "❌ No adapter found! Training may have failed."
            echo "Check: ssh lanta 'cat $BASE/outputs/20k-resume_${TRAIN_JOB}.err | tail -20'"
            exit 1
        fi
        
        echo "✅ Adapter found, submitting eval..."
        
        # Submit eval
        JOB_ID=$(ssh $REMOTE "cd $BASE && sbatch scripts/eval.sh models/Qwen3.5-9B outputs/clean-qwen35-20k qwen3_5 clean-qwen35-20k" 2>&1 | grep -oP '\d+$')
        
        echo "📤 Eval submitted! Job ID: $JOB_ID"
        echo ""
        echo "Next steps:"
        echo "  1. Monitor: ssh lanta 'squeue -j $JOB_ID'"
        echo "  2. When done: bash scripts/sync_and_f1.sh"
        exit 0
    fi
    
    # Show progress
    PROGRESS=$(ssh $REMOTE "tail -1 $BASE/outputs/clean-qwen35-20k/trainer_log.jsonl 2>/dev/null" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(f\"Step {d['current_steps']}/{d['total_steps']} ({d['percentage']:.0f}%) | Loss: {d['loss']:.4f} | ETA: {d['remaining_time']}\")
except: print('Loading...')
" 2>/dev/null)
    
    echo "  [$(date +%H:%M)] Job $TRAIN_JOB: $STATUS | $PROGRESS"
    sleep 120
done

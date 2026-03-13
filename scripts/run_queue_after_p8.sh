#!/bin/bash
# Queue: runs AFTER P8 finishes. Fastest jobs first.
set -e
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== Waiting for P8 to finish ==="
while pgrep -f run_all_combined > /dev/null 2>&1; do sleep 30; done
echo "=== P8 done. Starting additional experiments ==="

echo ""
echo "########## P23: Edge Quantization (~2hr) ##########"
python3 /workspace/run_p23_quant.py

echo ""
echo "########## P9: SFT vs DPO (~3hr) ##########"
python3 /workspace/run_p9_alignment.py

echo ""
echo "########## P18: Zero-Shot (~4hr) ##########"
python3 /workspace/run_p18_zeroshot.py

echo ""
echo "########## P14: LoRA vs OFT (~6hr) ##########"
python3 /workspace/run_p14_lora_vs_oft.py

echo ""
echo "########## P20: Cross-Domain (~6hr) ##########"
python3 /workspace/run_p20_crossdomain.py

echo ""
echo "========== ALL EXPERIMENTS COMPLETE =========="
echo "Results:"
ls -la /workspace/results/
cat /workspace/results/*.json

echo "=== AUTO-STOPPING INSTANCE ==="
sleep 10
poweroff

#!/bin/bash
# Run all fixed experiments sequentially
# Estimated total: ~4 hours on RTX 4090
set -e

cd /workspace
mkdir -p results

echo "============================================"
echo "  STARTING ALL FIXED EXPERIMENTS"
echo "  $(date)"
echo "============================================"

# 1. P18: Zero-shot (fixed prompt — no label leak)
echo ""
echo ">>> [1/5] P18: Zero-shot generalization (fixed prompt)"
python3 run_p18_zeroshot.py 2>&1 | tee results/p18_fixed.log
echo ">>> P18 DONE at $(date)"

# 2. P9: SFT + DPO (reimplemented with proper DPO)
echo ""
echo ">>> [2/5] P9: SFT vs DPO (proper DPO implementation)"
python3 run_p9_alignment.py 2>&1 | tee results/p9_fixed.log
echo ">>> P9 DONE at $(date)"

# 3. P14: LoRA vs OFT (fixed OFT + 5 seeds)
echo ""
echo ">>> [3/5] P14: LoRA vs OFT (5 seeds)"
python3 run_p14_lora_vs_oft.py 2>&1 | tee results/p14_fixed.log
echo ">>> P14 DONE at $(date)"

# 4. P23: Quantization (fixed VRAM measurement)
echo ""
echo ">>> [4/5] P23: Edge Quantization (fixed VRAM reset)"
python3 run_p23_quant.py 2>&1 | tee results/p23_fixed.log
echo ">>> P23 DONE at $(date)"

# 5. P20: Cross-domain (with lenient F1)
echo ""
echo ">>> [5/5] P20: Cross-domain transfer (with lenient F1)"
python3 run_p20_crossdomain.py 2>&1 | tee results/p20_fixed.log
echo ">>> P20 DONE at $(date)"

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS DONE at $(date)"
echo "============================================"
echo ""
echo "Results:"
ls -la results/*.json

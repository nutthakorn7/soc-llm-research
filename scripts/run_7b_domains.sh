#!/bin/bash
# Run Qwen2.5-7B on all 3 domains
set -e

echo "=== Installing deps ==="
pip install torch transformers==4.44.0 datasets peft bitsandbytes accelerate trl==0.7.4 scikit-learn
echo "Setup done!"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')" 2>/dev/null || true

echo ""
echo "=== AG News — Qwen2.5-7B ==="
python3 train_crossdomain.py --domain ag_news --model Qwen/Qwen2.5-7B-Instruct --batch 2 --grad_acc 16

echo ""
echo "=== GoEmotions — Qwen2.5-7B ==="
python3 train_crossdomain.py --domain go_emotions --model Qwen/Qwen2.5-7B-Instruct --batch 2 --grad_acc 16

echo ""
echo "=== LedGAR — Qwen2.5-7B ==="
python3 train_crossdomain.py --domain ledgar --model Qwen/Qwen2.5-7B-Instruct --batch 2 --grad_acc 16

echo ""
echo "=== ALL 7B DONE ==="
cat /workspace/results/*.json

echo ""
echo "=== AUTO-STOPPING INSTANCE ==="
sleep 5
poweroff

#!/bin/bash
set -e
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== Installing deps ==="
pip install -q datasets peft bitsandbytes accelerate scikit-learn huggingface_hub==0.25.0
echo "Setup done!"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=== AG News — 7B ==="
python3 train_crossdomain.py --domain ag_news --model Qwen/Qwen2.5-7B-Instruct --batch 2 --grad_acc 16

echo ""
echo "=== GoEmotions — 7B ==="
python3 train_crossdomain.py --domain go_emotions --model Qwen/Qwen2.5-7B-Instruct --batch 2 --grad_acc 16

echo ""
echo "=== LedGAR — 7B ==="
python3 train_crossdomain.py --domain ledgar --model Qwen/Qwen2.5-7B-Instruct --batch 2 --grad_acc 16

echo ""
echo "=== AG News — 0.5B ==="
python3 train_crossdomain.py --domain ag_news --model Qwen/Qwen2.5-0.5B-Instruct --batch 4 --grad_acc 8

echo ""
echo "=== GoEmotions — 0.5B ==="
python3 train_crossdomain.py --domain go_emotions --model Qwen/Qwen2.5-0.5B-Instruct --batch 4 --grad_acc 8

echo ""
echo "=== LedGAR — 0.5B ==="
python3 train_crossdomain.py --domain ledgar --model Qwen/Qwen2.5-0.5B-Instruct --batch 4 --grad_acc 8

echo ""
echo "=== ALL DONE ==="
echo "Results:"
cat /workspace/results/*.json

echo "=== AUTO-STOPPING ==="
sleep 5
poweroff

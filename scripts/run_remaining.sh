#!/bin/bash
# Run ONLY the experiments that crashed/didn't run on previous Vast.ai session
# AG News 7B already completed — skip it
set -e
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== Installing deps ==="
pip install -q datasets peft bitsandbytes accelerate scikit-learn huggingface_hub==0.25.0
echo "Setup done!"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "########## Cross-domain: GoEmotions 7B (was crashed) ##########"
python3 train_crossdomain.py --domain go_emotions --model Qwen/Qwen2.5-7B-Instruct --batch 2 --grad_acc 16

echo ""
echo "########## Cross-domain: LedGAR 7B ##########"
python3 train_crossdomain.py --domain ledgar --model Qwen/Qwen2.5-7B-Instruct --batch 2 --grad_acc 16

echo ""
echo "########## Cross-domain: AG News 0.5B ##########"
python3 train_crossdomain.py --domain ag_news --model Qwen/Qwen2.5-0.5B-Instruct --batch 4 --grad_acc 8

echo ""
echo "########## Cross-domain: GoEmotions 0.5B ##########"
python3 train_crossdomain.py --domain go_emotions --model Qwen/Qwen2.5-0.5B-Instruct --batch 4 --grad_acc 8

echo ""
echo "########## Cross-domain: LedGAR 0.5B ##########"
python3 train_crossdomain.py --domain ledgar --model Qwen/Qwen2.5-0.5B-Instruct --batch 4 --grad_acc 8

echo ""
echo "########## P20: Cross-Domain Transfer (remaining) ##########"
python3 run_p20_crossdomain.py

echo ""
echo "=== ALL REMAINING DONE ==="
echo "Results:"
ls -la /workspace/results/
cat /workspace/results/*.json

echo "=== AUTO-STOPPING ==="
sleep 5
poweroff

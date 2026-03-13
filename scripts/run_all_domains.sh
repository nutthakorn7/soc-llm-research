#!/bin/bash
# Run all 3 domains on Vast.ai — total ~2-3 hours
set -e

echo "=== Installing deps ==="
bash vastai_setup.sh

echo ""
echo "=== AG News (4 classes, H=2.00) ==="
python3 train_crossdomain.py --domain ag_news

echo ""
echo "=== GoEmotions (28 classes, H=3.75) ==="
python3 train_crossdomain.py --domain go_emotions

echo ""
echo "=== LedGAR (100 classes, H=6.16) ==="
python3 train_crossdomain.py --domain ledgar

echo ""
echo "=== ALL DONE ==="
echo "Results:"
cat /workspace/results/*.json

echo ""
echo "=== AUTO-STOPPING INSTANCE ==="
sleep 5
poweroff

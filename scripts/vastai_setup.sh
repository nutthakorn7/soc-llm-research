#!/bin/bash
pip install -q datasets peft bitsandbytes accelerate scikit-learn
echo "Setup done!"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

#!/bin/bash
# Setup conda environment for SOC Alert Fine-tuning on Lanta HPC
set -e

echo "=== Loading modules ==="
module load cuda/12.6 Mamba/23.11.0-0

echo "=== Creating conda environment ==="
mamba create -n soc-finetune python=3.11 -y
source activate soc-finetune

echo "=== Installing PyTorch + CUDA ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing training dependencies ==="
pip install transformers accelerate peft trl bitsandbytes
pip install datasets huggingface_hub sentencepiece protobuf
pip install scikit-learn pandas numpy jsonlines
pip install flash-attn --no-build-isolation

echo "=== Installing vLLM for evaluation ==="
pip install vllm

echo "=== Installing LlamaFactory ==="
cd /project/lt200473-ttctvs/workshop-pretrain/LlamaFactory
pip install -e ".[torch,metrics]"

echo "=== Verify installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo "=== Setup complete! ==="
echo "Activate with: module load cuda/12.6 Mamba/23.11.0-0 && source activate soc-finetune"

---
library_name: peft
license: other
base_model: /project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-9B
tags:
- base_model:adapter:/project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-9B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: scale-qwen35-10k
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# scale-qwen35-10k

This model is a fine-tuned version of [/project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-9B](https://huggingface.co//project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-9B) on the salad_10k dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0588

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- total_eval_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2
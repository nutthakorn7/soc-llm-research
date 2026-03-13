---
library_name: transformers
license: other
base_model: /project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-0.8B
tags:
- llama-factory
- full
- generated_from_trainer
model-index:
- name: full-ft-q08-5k
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# full-ft-q08-5k

This model is a fine-tuned version of [/project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-0.8B](https://huggingface.co//project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-0.8B) on the clean_5k dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 5.2.0
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2

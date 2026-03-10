---
library_name: peft
license: other
base_model: /project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-0.8B
tags:
- base_model:adapter:/project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-0.8B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen35-08b-lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen35-08b-lora

This model is a fine-tuned version of [/project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-0.8B](https://huggingface.co//project/lt200473-ttctvs/soc-finetune/models/Qwen3.5-0.8B) on the salad_50k dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0586

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
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.0599        | 0.64   | 500  | 0.0595          |
| 0.0595        | 1.2790 | 1000 | 0.0590          |
| 0.0588        | 1.9190 | 1500 | 0.0589          |
| 0.0586        | 2.5581 | 2000 | 0.0587          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2
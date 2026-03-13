# SOC-LLM Research Papers

Unified research program on LLM fine-tuning for SOC alert classification.

## Narrative: "Strict F1 reveals what normalized F1 hides"

All papers share one discovery: fine-tuned LLMs **understand** cybersecurity concepts perfectly but **hallucinate label names** from pre-training. The `strict vs normalized F1` dual-metric framework exposes this gap.

## Papers

### Core Findings
| Dir | Title | Status |
|-----|-------|--------|
| `p3-soc-ft/` | Mind the Label Gap (main paper) | ✅ Draft |
| `p19-rule-of-law/` | Beyond Accuracy: 30-Item Checklist | ✅ Draft |
| `p24-cyber-datasets/` | Beyond SALAD: Difficulty Grading | ✅ Draft |

### Ablation Studies
| Dir | Title | Status |
|-----|-------|--------|
| `p6-scaling/` | 1,000 Labels Is All You Need | ✅ Draft |
| `p22-lora-rank/` | Higher Rank, More Hallucination | ✅ Draft |
| `p21-sub-1b/` | Bigger Models Follow Instructions | ✅ Draft |
| `p7-cost-efficient/` | $0.60 Is All You Need | ✅ Draft |

### Cross-Domain & Transfer
| Dir | Title | Status |
|-----|-------|--------|
| `p8-task-complexity/` | Entropy Predicts LLM Necessity | ⏳ DS evals |
| `p20-general-ai/` | SOC to Legal Contracts | ⏳ DS evals |
| `p18-zero-shot/` | Zero-Shot Transfer | ⏳ ZS evals |
| `p5-cascade/` | Entropy-Aware Cascade | ⏳ DS evals |

### Advanced Methods
| Dir | Title | Status |
|-----|-------|--------|
| `p9-rlhf-dpo/` | Alignment vs Hallucination | ⏳ DPO train |
| `p14-oft-vs-lora/` | OFT vs LoRA | ⏳ OFT eval |
| `p15-multi-task/` | One Model, Three Tasks | ✅ Draft |
| `p23-edge-quant/` | Quantize and Deploy | ⏳ 8-bit eval |

### Dataset & Multilingual
| Dir | Title | Status |
|-----|-------|--------|
| `p13-salad-v2/` | SALAD-v2 Multi-Source | ⏳ v2 eval |
| `p11-multilingual/` | Thai-Language SOC Alerts | ⏳ Thai eval |

## Quick Start
```bash
# Submit remaining evals (after queue reset)
ssh lanta 'bash /project/lt200473-ttctvs/soc-finetune/scripts/submit_remaining.sh'
```

## Key Numbers (verified)
- 7 models benchmarked (0.8B–9B)
- Strict F1 range: 46.1%–100% (Attack Category)
- Seed variance: 0.261–0.836 on identical config
- Cross-domain transfer: 0%
- Perfect compliance: reasoning models OR 20K training

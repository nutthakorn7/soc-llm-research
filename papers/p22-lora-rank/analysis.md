# P22: LoRA Rank Sensitivity Across Model Sizes

## Thesis
Optimal LoRA rank varies with model size. Small models overfit at high ranks; large models benefit from increased rank.

## Existing Data (Qwen3.5-0.8B)

| Rank | Avg F1 | Δ vs 64 |
|---|---|---|
| 16 | 99.5% | -0.5% |
| 32 | 88.7% | -11.3% |
| **64** | **100.0%** | baseline |
| 128 | 87.4% | **-12.6%** |

### Key Finding
> **Rank 128 WORSE than Rank 16!** Severe overfitting in 0.8B model.

## Experiment Design (NEW — submit tonight)

### Rank × Model Matrix

| | Rank 16 | Rank 32 | Rank 64 | Rank 128 |
|---|---|---|---|---|
| Qwen3.5-0.8B | ✅ 99.5% | ✅ 88.7% | ✅ 100% | ✅ 87.4% |
| SmolLM2-1.7B | ❌ | ❌ | ✅ 100% | ❌ |
| Phi-4-mini-3.8B | ❌ | ❌ | ✅ 100% | ❌ |
| DeepSeek-7B | ❌ | ❌ | ✅ 100% | ❌ |
| Qwen3-8B | ❌ | ❌ | ✅ 99.97% | ❌ |

### Expected Figure
```
F1 ↑
100% │  ●────●          ●────●  (7B+)
     │ /      \        /      
 95% │/        \      /        
     │          \    /         
 90% │           \  / (0.8B)   
     │            ○            
 85% │                         
     └──────────────────────── Rank →
      16   32   64   128
```

### Hypothesis
> Optimal rank ∝ sqrt(model_size). Small model = low rank sufficient.

## TODO: 12 training jobs (4 models × rank 16,32,128)
## Target: EACL / NAACL

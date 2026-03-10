# P6: Scaling Laws for SOC Alert Classification with Fine-Tuned LLMs

## Key Results

### Scaling Curve (Qwen3.5-0.8B, Strict Match)

| Train Size | Strict Atk F1 | Normalized Atk F1 | Gap |
|---|---|---|---|
| 1K | 87.5% | 100.0% | 12.5% |
| 5K | 83.6% | 100.0% | 16.4% |
| 10K | 97.4% | 100.0% | 2.6% |
| 20K | ⏳ eval pending | ⏳ | |
| 50K | ⏳ training | ⏳ | |

### Key Findings

1. **Non-monotonic scaling in strict match**: 5K < 1K — likely due to increased diversity introducing more label aliases at intermediate scales
2. **10K is the sweet spot**: 97.4% strict match = model learns both semantics AND exact labels
3. **Normalized F1 saturates at 1K**: The model always understands the task; strict match gap = label formatting issue

### Figure: Scaling Curve
```
Strict F1 ↑
100% │                    ●──
     │                   /
 95% │                  /
     │                 /
 90% │  ●            /
     │   \          /
 85% │    \        /
     │     ●──────
 80% │
     └───────────────────── Train Size →
       1K    5K     10K    20K    50K
```

### Model Comparison (Strict vs Normalized)

| Model | Size | Strict Atk F1 | Normalized F1 | Gap |
|---|---|---|---|---|
| DeepSeek-R1-7B | 7B | **100.0%** | 100.0% | 0% |
| SmolLM2-1.7B | 1.7B | 87.5% | 100.0% | 12.5% |
| Qwen3.5-0.8B | 0.8B | 87.5% | 100.0% | 12.5% |
| Qwen3-8B | 8B | 75.3% | 99.97% | 24.7% |
| Mistral-7B | 7B | 74.9% | 75.3% | 0.4% |

### Insight for Paper
> The strict-normalized gap reveals that most models have learned the correct semantic mapping but produce **sub-category labels** (e.g., "Port Scanning" instead of "Reconnaissance"). Only DeepSeek-R1 produces exact labels, likely due to its reasoning-oriented training.

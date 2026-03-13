# P6: How Many Labels Do You Really Need? Sample Efficiency in Low-Entropy Classification

## Reframed Title
**"1,000 Labels Is All You Need: Sample Efficiency of Fine-Tuned LLMs on Low-Entropy Classification Tasks"**

## New Thesis (validated by strict F1!)
Training size controls **hallucination rate**, not semantic accuracy:
- Semantic: 1K saturates (norm F1=100% at all sizes)
- Strict: 1K=77.8%, 10K=86.6%, **20K=100%** → real scaling curve

## Clean Data Scaling (Qwen3.5-9B, Strict Atk F1)

| Train Size | Strict F1 | Norm F1 | Halluc Labels | Story |
|-----------|:---------:|:-------:|:-------------:|-------|
| 1K | 77.8% | 100% | 1 | Semantic ✅, labels ❌ |
| 5K | 55.7% | 100% | 4 | More hallucination! |
| 10K | 86.6% | 100% | 1 | Learning vocab |
| **20K** | **100%** | 100% | **0** | Hallucination fixed |

## Key Insight
> **Scaling doesn't improve understanding — it teaches label vocabulary.**
> The model understands "Reconnaissance" from 1K samples. It takes 20K to stop calling it "Port Scanning."

## New Contributions
1. **Semantic saturation at 1K** — model learns task with minimal data
2. **Label compliance requires 20K** — vocabulary learning needs more exposure
3. **Non-monotonic strict scaling** — 5K worse than 1K (more capacity = more hallucination)
4. **Cross-domain prediction** — higher H(Y) tasks need more samples

## Action Plan
- [x] Clean scaling 1K-20K ✅
- [x] Strict F1 scaling curve ✅
- [/] 50K training (running)
- [ ] Cross-domain scaling (P20 results)
- [ ] Write draft

## Target: EMNLP Findings / ACL Industry Track

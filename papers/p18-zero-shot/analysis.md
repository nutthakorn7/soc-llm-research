# P18: Zero-Shot Transfer Across Attack Categories

## Experiment Design

Leave-one-category-out: Train on 7 categories, test on unseen 8th.

### Training Models (8 folds × 2 sizes)

| Held-out Category | 0.8B Status | 9B Status |
|---|---|---|
| Analysis | ✅ | ✅ zs9b-no-analysis |
| Backdoor | ✅ | ✅ zs9b-no-backdoor |
| Benign | ✅ | ✅ zs9b-no-benign |
| DoS | ✅ | ✅ zs9b-no-dos |
| Exploits | ✅ | ✅ zs9b-no-exploits |
| Fuzzers | ✅ | ✅ zs9b-no-fuzzers |
| Generic | ✅ | ✅ zs9b-no-generic |
| Reconnaissance | ✅ | ✅ zs9b-no-reconnaissance |

**All 16 models trained!** Eval running overnight.

## Expected Transfer Matrix

```
            Predicted Category →
True ↓      Recon  DoS  Expl  Fuzz  Anal  Back  Gene  Beni
Recon        —     ?    ?     ?     ?     ?     ?     ?
DoS          ?     —    ?     ?     ?     ?     ?     ?
Exploits     ?     ?    —     ?     ?     ?     ?     ?
Fuzzers      ?     ?    ?     —     ?     ?     ?     ?
Analysis     ?     ?    ?     ?     —     ?     ?     ?
Backdoor     ?     ?    ?     ?     ?     —     ?     ?
Generic      ?     ?    ?     ?     ?     ?     —     ?
Benign       ?     ?    ?     ?     ?     ?     ?     —
```

## Key Questions

1. Can a model identify "Reconnaissance" if never trained on it?
2. Which categories transfer well? (semantically similar = easier)
3. Does 9B outperform 0.8B on zero-shot transfer?

## Hypotheses

- DoS ↔ DDoS: High transfer (same family)
- Backdoor ↔ Exploits: Moderate (both post-exploitation)  
- Generic → everything: Low transfer (too vague)
- 9B > 0.8B for zero-shot: Larger models = better generalization

---
description: Validation protocol when any metric reports >=99% F1 — run before trusting or publishing high scores
---

# Perfect Score Validation Protocol

When **any** model reports F1 ≥ 99%, run this checklist before trusting the result.

## Step 1: Data Integrity

1. Verify zero train/test overlap:
```bash
python3 -c "
import json
train = set(json.dumps(x['instruction']) for x in json.load(open('train.json')))
test = set(json.dumps(x['instruction']) for x in json.load(open('test.json')))
overlap = train & test
print(f'Train: {len(train)} | Test: {len(test)} | Overlap: {len(overlap)}')
if overlap: print('⚠️  DATA LEAKAGE DETECTED')
else: print('✅ Zero overlap')
"
```

// turbo
2. Check test set size and class distribution:
```bash
python3 -c "
import json
from collections import Counter
data = json.load(open('test.json'))
labels = [x.get('output','') for x in data]
dist = Counter(labels)
print(f'Test samples: {len(data)} | Classes: {len(dist)}')
for k,v in dist.most_common(): print(f'  {k}: {v} ({100*v/len(data):.1f}%)')
"
```

## Step 2: Prediction Audit

3. Manually inspect 20 random predictions from the perfect-scoring model:
   - Do predictions look like memorized outputs or genuine classification?
   - Are there suspiciously identical response patterns?
   - Do edge cases (rare classes) get correct answers?

// turbo
4. Check prediction diversity:
```bash
python3 -c "
import json
from collections import Counter
preds = json.load(open('predictions.json'))  # adjust filename
outputs = [p.get('predict','') for p in preds]
dist = Counter(outputs)
print(f'Unique outputs: {len(dist)} / {len(outputs)}')
for k,v in dist.most_common(10): print(f'  {k}: {v}')
"
```

## Step 3: Cross-Seed Stability

5. Check if the model achieves 100% across multiple seeds:
   - 100% on 1 seed but not others → seed-specific artifact, report with caveat
   - 100% on ALL seeds → more likely genuine
   - Report seed variance in the paper

## Step 4: Comparison Sanity

6. Verify the perfect score model against known-difficult samples:
   - If DT also gets 100% → task is trivially separable, LLM adds no value
   - If DT gets <90% but LLM gets 100% → LLM is genuinely better
   - If ALL models get 100% → test set may be too easy

## Step 5: Report Requirements

7. If you decide the score is genuine, the paper MUST:
   - State which seeds achieve 100% and which don't
   - Report confidence intervals or seed variance
   - Acknowledge the low-entropy nature of the task
   - Compare against trivial baselines (DT, majority class)
   - NOT claim "state-of-the-art" without cross-domain validation

## Red Flags (STOP and investigate)

- Train/test overlap > 0
- Test set < 100 samples
- Only 1 class in test set
- Model outputs identical strings for all inputs
- 100% on ALL models (task too easy)

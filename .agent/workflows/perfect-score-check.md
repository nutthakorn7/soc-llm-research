---
description: Validation protocol when any metric reports ≥99% F1 — run before trusting or publishing high scores
---

# Perfect Score Suspicion Protocol

> เมื่อ F1 ≥ 99% = สงสัยก่อนเชื่อ เสมอ
> Protocol นี้ใช้ได้กับทุก paper ในโปรเจค

---

## Level 1: Sanity Checks (5 นาที, ไม่ต้องเขียน code)

ตอบคำถาม 4 ข้อ:
1. **H(Y) < 1 bit?** → Task trivially separable (เช่น SALAD Classification H=0.083)
2. **Majority class > 95%?** → F1 inflated (เช่น 99.9% Malicious)
3. **DT/SVM baseline ก็ได้ ≥99%?** → LLM ไม่จำเป็น
4. **Unique input patterns < 1000?** → Model แค่จำ lookup table (เช่น SALAD = 87 patterns)

ถ้าตอบ "ใช่" ≥ 2 ข้อ → ⚠️ reframe narrative, ไม่ควร claim 100% เป็น contribution หลัก

---

## Level 2: Metric Verification (30 นาที)

### L2.1 — Strict F1 vs Normalized F1

// turbo
```bash
cd /Users/pop7/Code/Lanta && python3 -c "
import json, re
from sklearn.metrics import f1_score

PREDICTION_FILE = 'results/eval-clean-qwen35-5k/generated_predictions.jsonl'

with open(PREDICTION_FILE) as f:
    data = [json.loads(l) for l in f]

# Extract Attack Category (raw, no aliases)
true_labels = []
pred_labels = []
for d in data:
    m_t = re.search(r'Attack Category:\s*(.+)', d['label'])
    m_p = re.search(r'Attack Category:\s*(.+)', d['predict'])
    if m_t and m_p:
        true_labels.append(m_t.group(1).strip())
        pred_labels.append(m_p.group(1).strip())

strict_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
print(f'Strict F1 (NO aliases): {strict_f1:.4f}')
print(f'Exact match: {sum(1 for t,p in zip(true_labels,pred_labels) if t==p)}/{len(true_labels)}')
print()

# Compare with normalized (using our aliases)
ALIASES = {
    'port scanning': 'reconnaissance', 'scanning': 'reconnaissance',
    'backdoors': 'backdoor', 'shellcode': 'backdoor',
    'bots': 'backdoor', 'worms': 'backdoor',
    'ddos': 'dos', 'denial of service': 'dos',
    'l2tp': 'reconnaissance',
}
norm_true = [ALIASES.get(t.lower(), t.lower()) for t in true_labels]
norm_pred = [ALIASES.get(p.lower(), p.lower()) for p in pred_labels]
norm_f1 = f1_score(norm_true, norm_pred, average='macro', zero_division=0)
print(f'Normalized F1 (with aliases): {norm_f1:.4f}')
print(f'Gap: {norm_f1 - strict_f1:.4f}')

if norm_f1 - strict_f1 > 0.10:
    print('🚨 RED FLAG: Normalized-Strict gap > 10%!')
else:
    print('✅ Gap acceptable')
"
```

⚠️ **เปลี่ยน PREDICTION_FILE ตาม eval ที่ต้องการตรวจ**

### L2.2 — Random Baseline

// turbo
```bash
cd /Users/pop7/Code/Lanta && python3 -c "
import json, re, numpy as np
from sklearn.metrics import f1_score
from collections import Counter

PREDICTION_FILE = 'results/eval-clean-qwen35-5k/generated_predictions.jsonl'

with open(PREDICTION_FILE) as f:
    data = [json.loads(l) for l in f]

# Attack Category
true = [re.search(r'Attack Category:\s*(.+)', d['label']).group(1).strip() for d in data if re.search(r'Attack Category:', d['label'])]
classes = list(set(true))
probs = [true.count(c)/len(true) for c in classes]

np.random.seed(42)
random_f1s = [f1_score(true, np.random.choice(classes, len(true), p=probs), average='macro', zero_division=0) for _ in range(1000)]
print(f'Random baseline F1: {np.mean(random_f1s):.4f} ± {np.std(random_f1s):.4f}')
print(f'Majority vote accuracy: {max(Counter(true).values())/len(true)*100:.1f}%')

# Classification
cls_true = [re.search(r'Classification:\s*(.+)', d['label']).group(1).strip() for d in data if re.search(r'Classification:', d['label'])]
print(f'Classification majority: {max(Counter(cls_true).values())/len(cls_true)*100:.1f}%')
if max(Counter(cls_true).values())/len(cls_true) > 0.95:
    print('⚠️ Classification >95% single class — metric not informative')
"
```

### L2.3 — Manual Spot-Check (20 samples)

// turbo
```bash
cd /Users/pop7/Code/Lanta && python3 -c "
import json, random
random.seed(42)

PREDICTION_FILE = 'results/eval-clean-qwen35-5k/generated_predictions.jsonl'

with open(PREDICTION_FILE) as f:
    lines = [json.loads(l) for l in f]

indices = random.sample(range(len(lines)), 20)
matches = 0
for i in indices:
    d = lines[i]
    label = d['label'].strip()[:150]
    predict = d['predict'].strip()[:150]
    match = '✅' if d['label'].strip() == d['predict'].strip() else '❌'
    if d['label'].strip() == d['predict'].strip(): matches += 1
    print(f'{match} #{i}: L={label}')
    print(f'         P={predict}')
    print()

pct = matches/20*100
print(f'Spot-check: {matches}/20 exact match ({pct:.0f}%)')
if pct < 80:
    print('🚨 RED FLAG: Exact match < 80% in spot-check!')
"
```

### L2.4 — sklearn Cross-Verify

// turbo
```bash
cd /Users/pop7/Code/Lanta && python3 -c "
import json, re
from sklearn.metrics import classification_report, f1_score

PREDICTION_FILE = 'results/eval-clean-qwen35-5k/generated_predictions.jsonl'

with open(PREDICTION_FILE) as f:
    data = [json.loads(l) for l in f]

for task, field in [('Classification', 'Classification'), ('Triage', 'Triage Decision'), ('Attack Category', 'Attack Category')]:
    true = [re.search(rf'{field}:\s*(.+)', d['label']).group(1).strip() for d in data if re.search(rf'{field}:', d['label'])]
    pred = [re.search(rf'{field}:\s*(.+)', d['predict']).group(1).strip() for d in data if re.search(rf'{field}:', d['predict'])]
    if len(true) == len(pred):
        f1 = f1_score(true, pred, average='macro', zero_division=0)
        print(f'--- {task}: sklearn macro-F1 = {f1:.4f} ---')
        if f1 < 0.95:
            print(classification_report(true, pred, zero_division=0))
    else:
        print(f'--- {task}: MISMATCH true={len(true)} pred={len(pred)} ---')
"
```

---

## Level 3: Deep Inspection (ก่อน submit paper)

### L3.1 — Per-Class F1 Table

// turbo
```bash
cd /Users/pop7/Code/Lanta && python3 -c "
import json, re
from sklearn.metrics import classification_report

PREDICTION_FILE = 'results/eval-clean-qwen35-5k/generated_predictions.jsonl'

with open(PREDICTION_FILE) as f:
    data = [json.loads(l) for l in f]

true = [re.search(r'Attack Category:\s*(.+)', d['label']).group(1).strip() for d in data]
pred = [re.search(r'Attack Category:\s*(.+)', d['predict']).group(1).strip() for d in data]
print(classification_report(true, pred, zero_division=0, digits=4))
"
```

### L3.2 — Confusion Matrix

// turbo
```bash
cd /Users/pop7/Code/Lanta && python3 -c "
import json, re
from sklearn.metrics import confusion_matrix
import numpy as np

PREDICTION_FILE = 'results/eval-clean-qwen35-5k/generated_predictions.jsonl'

with open(PREDICTION_FILE) as f:
    data = [json.loads(l) for l in f]

true = [re.search(r'Attack Category:\s*(.+)', d['label']).group(1).strip() for d in data]
pred = [re.search(r'Attack Category:\s*(.+)', d['predict']).group(1).strip() for d in data]
labels = sorted(set(true) | set(pred))
cm = confusion_matrix(true, pred, labels=labels)
print(f'{'':>20}', '  '.join(f'{l[:6]:>6}' for l in labels))
for i, row in enumerate(cm):
    nonzero = [f'{labels[j][:6]}={v}' for j,v in enumerate(row) if v > 0 and j != i]
    print(f'{labels[i]:>20}', '  '.join(f'{v:>6}' for v in row), '  ', ', '.join(nonzero) if nonzero else '')
"
```

### L3.3 — Predicted Vocab Audit

// turbo
```bash
cd /Users/pop7/Code/Lanta && python3 -c "
import json, re
from collections import Counter

PREDICTION_FILE = 'results/eval-clean-qwen35-5k/generated_predictions.jsonl'

with open(PREDICTION_FILE) as f:
    data = [json.loads(l) for l in f]

true_vocab = Counter(re.search(r'Attack Category:\s*(.+)', d['label']).group(1).strip() for d in data)
pred_vocab = Counter(re.search(r'Attack Category:\s*(.+)', d['predict']).group(1).strip() for d in data)

print(f'TRUE vocab ({len(true_vocab)} labels): {sorted(true_vocab.keys())}')
print(f'PRED vocab ({len(pred_vocab)} labels): {sorted(pred_vocab.keys())}')
print()

hallucinated = set(pred_vocab) - set(true_vocab)
if hallucinated:
    print(f'⚠️ HALLUCINATED LABELS ({len(hallucinated)}):')
    for h in sorted(hallucinated, key=lambda x: -pred_vocab[x]):
        # Find which true label this replaces
        replacements = Counter()
        for d in data:
            m_p = re.search(r'Attack Category:\s*(.+)', d['predict'])
            m_t = re.search(r'Attack Category:\s*(.+)', d['label'])
            if m_p and m_t and m_p.group(1).strip() == h:
                replacements[m_t.group(1).strip()] += 1
        print(f'  \"{h}\" ({pred_vocab[h]}×) — replaces: {dict(replacements)}')
else:
    print('✅ No hallucinated labels')
"
```

### L3.4 — Train/Test Overlap Recheck

// turbo
```bash
cd /Users/pop7/Code/Lanta && python3 -c "
import json

train = json.load(open('data/train_5k_clean.json'))
test = json.load(open('data/test_held_out.json'))

train_inputs = set(d['input'] for d in train)
test_inputs = set(d['input'] for d in test)

overlap = train_inputs & test_inputs
print(f'Train unique inputs: {len(train_inputs)}')
print(f'Test unique inputs: {len(test_inputs)}')
print(f'Overlap: {len(overlap)}')
if overlap:
    print('🚨 DATA LEAKAGE DETECTED!')
    for s in list(overlap)[:3]:
        print(f'  Example: {s[:100]}...')
else:
    print('✅ Zero overlap — clean split verified')
"
```

---

## Level 4: Publication Evidence

7. **Cross-domain**: ใช้ model เดียวกัน test บน dataset อื่น (AG News, GoEmotions, LedGAR)
8. **Multi-seed**: ≥3 seeds, report mean ± std (ใช้ seed 42, 77, 999 หรือ 42, 123, 2024)
9. **Save evidence**: เก็บ output ทุก Level ไว้ใน `results/validation/` folder

---

## Decision Matrix

| Strict F1 | Normalized F1 | DT F1 | Action |
|-----------|--------------|-------|--------|
| ≥99% | ≥99% | <80% | ✅ Genuine — publish strict F1 as primary |
| ≥99% | ≥99% | ≥99% | 🟡 Task too easy — reframe, emphasize H(Y) |
| 80-99% | ≥99% | <80% | ⚠️ Report BOTH prominently, explain alias |
| <80% | ≥99% | any | 🚨 Aliases hiding major errors — strict F1 is primary |
| <50% | ≥99% | any | 🚨🚨 Metric BROKEN — do NOT publish normalized |

---

## Quick Reference: Which Check Catches What

| Problem | Caught By |
|---------|-----------|
| Data leakage | L3.4 (overlap recheck) |
| Label aliasing inflation | L2.1 (strict vs normalized) |
| Hallucinated labels | L3.3 (vocab audit) |
| Rare class failure | L3.1 (per-class F1) |
| Trivial task | L1.1-L1.4 (sanity checks) |
| Custom code bug | L2.4 (sklearn cross-verify) |
| Metric floor issue | L2.2 (random baseline) |
| Format/extraction error | L2.3 (manual spot-check) |

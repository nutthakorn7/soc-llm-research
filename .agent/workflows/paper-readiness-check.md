---
description: Pre-paper validation checklist — run before starting any new paper to verify data sufficiency and research viability
---

# Paper Readiness Check

ใช้ก่อนเริ่มเขียนทุก paper เพื่อตรวจสอบว่า:
1. ควรทำเรื่องนี้อยู่ไหม (novelty + gap)
2. Data เรามีเพียงพอไหม

## Step 1: Literature Gap Check

Search ว่ามีใครทำเรื่องนี้แล้วหรือยัง:

```bash
# Search Google Scholar / Semantic Scholar สำหรับ keywords ของ paper
# ดูว่ามี prior work ไหม → ถ้ามีแล้ว ต้อง differentiate อย่างไร
```

Criteria:
- [ ] ไม่มี paper ที่ตรงเรื่องเดียวกัน 100%
- [ ] มี gap ที่เราจะ fill ได้ชัดเจน
- [ ] เราสามารถ cite P1 (SALAD) + P2 (TRUST-SOC) + P3 (SOC-FT) ได้

## Step 2: Data Sufficiency Check

// turbo
```bash
ssh lanta 'python3 -c "
import json
# Check data files exist
for f in [\"test_held_out.json\", \"val_held_out.json\", \"train_5k_clean.json\"]:
    path = f\"/project/lt200473-ttctvs/soc-finetune/data/{f}\"
    try:
        d = json.load(open(path))
        print(f\"  ✅ {f}: {len(d)} samples\")
    except: print(f\"  ❌ {f}: NOT FOUND\")
"'
```

Check:
- [ ] `clean_test` (test_held_out.json) มีจำนวน samples เพียงพอ
- [ ] Attack categories ครอบคลุม (ปัจจุบัน test set มี 8/15)
- [ ] ไม่มี data leakage (ใช้ `clean_*` เท่านั้น)

## Step 3: Adapters & Eval Results Check

// turbo
```bash
ssh lanta 'echo "=== ADAPTERS ===" && for d in /project/lt200473-ttctvs/soc-finetune/outputs/*/adapter_config.json; do dirname "$d" | xargs basename; done 2>/dev/null | sort && echo "" && echo "=== EVAL PREDICTIONS ===" && for d in /project/lt200473-ttctvs/soc-finetune/outputs/eval-*/generated_predictions.jsonl; do name=$(dirname "$d" | xargs basename); lines=$(wc -l < "$d"); echo "  $name: $lines"; done 2>/dev/null | sort'
```

Check:
- [ ] Adapter ที่ต้องการมีอยู่ (trained with `clean_*`)
- [ ] Eval predictions มีอยู่ (run on `clean_test`)
- [ ] ผล F1 ถูกคำนวณแล้ว

## Step 3b: Pre-Flight Eval Check ⭐ NEW

ก่อน submit batch eval jobs ให้เช็คเสมอว่า:
- [ ] Script ใช้ **test dataset ที่ถูกต้อง** (ไม่ hardcode `clean_test`)
- [ ] ใช้ `flexible_eval.sh` สำหรับ cross-domain / zero-shot evals
- [ ] Verify จาก first prediction ว่า output domain ตรงกับ test dataset

```bash
# ตรวจว่า eval script ใช้ test dataset ไหน
grep "eval_dataset" /project/lt200473-ttctvs/soc-finetune/scripts/eval.sh
# ถ้าเจอ hardcoded → ใช้ flexible_eval.sh แทน
```

> ⚠️ บทเรียน: 26 GPU-hours เสียเปล่าเพราะ eval.sh hardcode `--eval_dataset clean_test`

## Step 4: Clustering / Feature Analysis Validation

// turbo
```bash
cat /Users/pop7/Code/Lanta/results/paper_results/clustering_analysis.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
fi = d.get('feature_importance', {})
pa = d.get('pattern_analysis', {})
km = d.get('kmeans_vs_attack_category', {})
print('=== DATA CHARACTERISTICS ===')
print(f'  Unique patterns: {pa.get(\"total_unique_patterns\", \"?\")}')
print(f'  Ambiguous: {pa.get(\"ambiguous_patterns\", \"?\")}')
print(f'  K-Means ARI (vs Attack Cat): {km.get(\"ari\", \"?\")}')
print(f'  Silhouette: {km.get(\"silhouette\", \"?\")}')
print()
print('=== FEATURE IMPORTANCE (MI for Attack Category) ===')
for feat, vals in fi.items():
    mi = vals.get('mi_attack_category', 0)
    flag = '⚠️ ZERO' if mi == 0 else ''
    print(f'  {feat:<25} {mi:.4f} {flag}')
"
```

ถ้า paper เกี่ยวกับ feature analysis ต้องตระหนักว่า:
- SALAD มีแค่ 87 unique patterns ใน test set (870 total)
- Zero ambiguity → ทุก pattern map ไป label เดียว
- Network Segment ไม่มี information gain (MI = 0.000)
- K-Means ARI = 0.91 → unsupervised เกือบเท่า supervised

## Step 5: Strict F1 Verification ⭐ NEW

เมื่อ eval เสร็จ ให้รัน `/perfect-score-check` ทุกครั้ง:
- [ ] Strict F1 = Normalized F1? (gap < 10%)
- [ ] Manual spot-check 20 samples
- [ ] Predicted vocab = True vocab (no hallucination)
- [ ] ถ้า strict ≠ normalized → report BOTH ใน paper

## Step 6: Go/No-Go Decision

| Criteria | Required |
|---|---|
| Literature gap exists | ✅ ต้องมี |
| Data files present & clean | ✅ ต้องมี |
| Adapters trained on clean data | ✅ ถ้า paper ต้องใช้ |
| Eval on correct test set done | ✅ ถ้า paper ต้องใช้ |
| Strict F1 verified | ✅ ต้องมี |
| Novelty ≥ ⭐⭐⭐ | ✅ ต้องมี |
| Correctness risk ≤ 🟡 | แนะนำ |

> ⚠️ IMPORTANT: ถ้า data ไม่พอ (เช่น test มีแค่ 8 attack categories) ต้องระบุเป็น limitation ใน paper ไม่ใช่ซ่อนไว้


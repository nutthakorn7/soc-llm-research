---
description: Pre-paper validation checklist — run before starting any new paper to verify data sufficiency and research viability
---

# Paper Readiness Check

Run this checklist on each paper to verify it has sufficient data and rigor for publication.

## Step 1: Data Sufficiency

// turbo
1. Check that all tables have real data (no `---` placeholders):
```bash
cd <paper_dir> && grep -n '\-\-\-' main.tex | grep -v 'comment\|%\|copyright\|---}'
```

2. Verify each claim in the abstract is backed by a table/figure/equation:
   - Extract all numeric claims from abstract
   - Cross-reference each with specific table/figure

3. Check that all figures are generated and exist:
```bash
cd <paper_dir> && grep 'includegraphics' main.tex | sed 's/.*{\(.*\)}/\1/' | while read f; do
  [ -f "$f" ] && echo "OK: $f" || echo "MISSING: $f"
done
```

## Step 2: Statistical Rigor

4. Check for statistical tests:
   - Multi-seed results: Are means ± std reported?
   - Comparisons: Are p-values or chi-squared tests present?
   - Single-seed results: Is this acknowledged as a limitation?

5. Check sample sizes:
   - Training set: ≥1K samples recommended
   - Test set: ≥100 samples (ideally ≥500)
   - Number of classes: reported and matches data

## Step 3: Reproducibility

6. Check that the paper specifies:
   - [ ] Hardware (GPU model, memory)
   - [ ] Software versions (framework, libraries)
   - [ ] Hyperparameters (LR, batch size, epochs, LoRA rank/alpha)
   - [ ] Random seeds used
   - [ ] Data split ratios
   - [ ] Code availability statement

## Step 4: Novelty Verification

7. Verify novelty claim is specific and defensible:
   - [ ] Novelty gap statement present in Related Work
   - [ ] At least 3 points of differentiation from prior work
   - [ ] Not just "first to apply X to Y" without methodology innovation

## Step 5: Citation Completeness

// turbo
8. Verify bibliography:
```bash
cd <paper_dir> && python3 -c "
import re
with open('main.tex') as f: c=f.read()
bibs=set(re.findall(r'\\\\bibitem\{([^}]+)\}',c))
cites=set()
for m in re.findall(r'\\\\cite\{([^}]+)\}',c):
    for k in m.split(','): cites.add(k.strip())
print(f'refs: {len(bibs)} | uncited: {len(bibs-cites)} | missing: {len(cites-bibs)}')
if bibs-cites: print(f'  Uncited: {sorted(bibs-cites)}')
if cites-bibs: print(f'  Missing: {sorted(cites-bibs)}')
# Check companion citations
companions = [k for k in cites if k.startswith('p')]
print(f'  Companion papers cited: {sorted(companions)}')
"
```

## Step 6: Structural Completeness

9. Check that the paper has all required sections:
   - [ ] Abstract with clear contributions
   - [ ] Introduction with motivation and contribution list
   - [ ] Related Work with novelty gap
   - [ ] Methodology/Experimental Setup
   - [ ] Results with tables/figures
   - [ ] Discussion with limitations
   - [ ] Conclusion with key findings, future work, broader impact
   - [ ] Acknowledgments
   - [ ] Complete bibliography

## Readiness Verdict

After completing all steps, assign a readiness score:

- **READY** (9-10/10): Ship it!
- **ALMOST** (7-8/10): Minor gaps, fixable in <1 hour
- **NOT READY** (≤6/10): Major gaps, needs more data/analysis

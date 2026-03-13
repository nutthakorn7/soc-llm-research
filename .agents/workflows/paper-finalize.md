---
description: Paper finalization — proofread, plagiarism check, humanize (run on every paper before submission)
---

# Paper Finalize Workflow

Run this workflow on every paper after content expansion is complete.
Repeat the cycle until no issues remain.

## Round 1: Proofread

// turbo
1. Compile the paper: `cd <paper_dir> && echo "" | pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "Output written"`

2. Read the full `main.tex` and check for:
   - **Duplicate `\usepackage{}`** imports
   - **Redundant paragraphs** (same content repeated in different sections)
   - **Incorrect counts** (e.g., "6 error types" but only 5 listed)
   - **Math errors** (cost calculations, percentages, sum mismatches)
   - **Inconsistent data** across tables, text, and abstract
   - **Broken refs/labels** (`\ref{}` pointing to undefined `\label{}`)
   - **Table vs Figure label mismatch** (e.g., `\label{fig:X}` on a table)
   - **Abstract numbers matching body** (model count, F1 ranges, costs)

3. Fix all issues found.

## Round 2: Plagiarism Check

4. Scan all sections for plagiarism risks:
   - **HIGH RISK**: Sentences that copy formula definitions verbatim from source papers (e.g., LoRA, QLoRA definitions)
   - **HIGH RISK**: Introduction statistics that appear identically across many papers
   - **MEDIUM RISK**: Related Work with formulaic "[Author] [verb]ed [title paraphrase]" patterns
   - **MEDIUM RISK**: Abstract ↔ Conclusion near-identical sentences (self-plagiarism)
   - **LOW RISK**: Methodology descriptions unique to this work

5. Rewrite all HIGH RISK passages. Note MEDIUM RISK in report.

## Round 3: Humanize

6. Check for AI-generated writing patterns:
   - **Overuse of filler**: "importantly", "notably", "significantly", "comprehensive", "crucial"
   - **Excessive hedging**: "it is worth noting that", "it should be mentioned that"
   - **Robotic transitions**: "Furthermore,", "Moreover,", "Additionally," at the start of every paragraph
   - **Repetitive sentence structure**: multiple consecutive "X does Y. Z does W." patterns
   - **Over-enthusiasm**: "groundbreaking", "revolutionary", "paradigm-shifting"
   - **Empty claims**: "to the best of our knowledge" (unless truly novel)

7. Rewrite flagged passages to sound more natural and human-written:
   - Vary sentence length and structure
   - Use active voice where appropriate
   - Replace filler words with specific claims
   - Start paragraphs with the actual content, not transition words

## Round 4: Final Verification

// turbo
8. Compile twice: `cd <paper_dir> && echo "" | pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "Output written" && echo "" | pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "Output written"`

// turbo
9. Run reference validation:
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
"
```

// turbo
10. Check for LaTeX warnings: `cd <paper_dir> && grep -c "undefined\|multiply" main.log`

11. Report final stats: pages, refs, figures, tables, equations, lines.

## If Any Issues Found

12. Go back to Round 1 and repeat until clean.

#!/usr/bin/env python3
"""Generate figures for P19: Beyond Accuracy Checklist."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3
})

# ── Figure 1: F1 Gap — Reported vs Actual ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
cases = ['CS1: Label\nAliasing', 'CS2: Data\nLeakage', 'CS3: ML\nvs LLM',
         'CS4: Rare\nClass', 'CS5: Class\nImbalance', 'CS6: Wrong\nDataset']
reported = [100, 66.1, 100, 91.7, 100, 95]
actual = [55.7, 100, 90.9, 0, 50, 0]
# CS2 is reversed: leaky said 66.1 but clean = 100
# CS3: LLM vs SVM
# CS4: per-class
# CS5: majority
# CS6: wrong dataset

x = np.arange(6)
w = 0.3
bars_r = ax.bar(x - w/2, reported, w, label='Reported / Misleading', color='#ef4444',
                edgecolor='black', linewidth=0.5, alpha=0.8)
bars_a = ax.bar(x + w/2, actual, w, label='Actual / Corrected', color='#16a34a',
                edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(cases, fontsize=7)
ax.set_ylabel('F1 (%)')
ax.set_ylim(0, 115)
ax.set_title('6 Case Studies: Reported vs Actual Performance')
ax.legend(loc='upper right', fontsize=8)

# Gap annotations
gaps = [44.3, 33.9, 9.1, 91.7, 50, 95]
for i, gap in enumerate(gaps):
    ax.annotate(f'{gap:.0f}pt gap', xy=(i, max(reported[i], actual[i]) + 2),
                fontsize=6, ha='center', color='darkred', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_gaps.pdf')
plt.close()
print('OK fig_gaps.pdf')

# ── Figure 2: Checklist Coverage by Category ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
categories = ['Dataset\nIntegrity', 'Baseline\nSelection', 'Metric\nSelection',
              'Statistical\nRigor', 'Label\nQuality', 'Ablation\nDesign',
              'Cost\nTransp.', 'Repro\nArtifacts', 'Perfect\nScore']
items = [5, 4, 4, 3, 4, 3, 3, 2, 2]
case_coverage = [3, 2, 3, 1, 4, 2, 1, 1, 1]  # case studies that test this
colors = ['#3b82f6', '#3b82f6', '#eab308', '#eab308', '#ef4444', '#ef4444',
          '#16a34a', '#16a34a', '#7c3aed']

bars = ax.bar(range(9), items, color=colors, width=0.6,
              edgecolor='black', linewidth=0.5, alpha=0.8)
ax.bar(range(9), case_coverage, color='none', width=0.6,
       edgecolor='black', linewidth=2, linestyle='--')

ax.set_xticks(range(9))
ax.set_xticklabels(categories, fontsize=7)
ax.set_ylabel('Number of Items')
ax.set_title('30-Item Checklist: Category Distribution')

for bar, n in zip(bars, items):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(n), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_checklist.pdf')
plt.close()
print('OK fig_checklist.pdf')

# ── Figure 3: Perfect Score Protocol Flow ──
fig, ax = plt.subplots(figsize=(5.5, 4.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Perfect Score Suspicion Protocol', fontsize=12, fontweight='bold')

levels = [
    ('Level 1\n(5 min)', 'Quick sanity checks:\nclass distribution,\nbaseline comparison', '#3b82f6', 8.5),
    ('Level 2\n(30 min)', 'Label audit:\npredicted vs true vocab,\nalias mapping review', '#eab308', 6.5),
    ('Level 3\n(2 hr)', 'Deep inspection:\nper-class F1, seed sweep,\ncross-validation', '#f97316', 4.5),
    ('Level 4\n(4 hr)', 'Adversarial test:\nperturbation, held-out\ndomain, archived proof', '#ef4444', 2.5),
]

for label, desc, color, y in levels:
    rect = plt.Rectangle((0.5, y-0.8), 3, 1.6, facecolor=color, alpha=0.3,
                          edgecolor=color, linewidth=2, clip_on=False)
    ax.add_patch(rect)
    ax.text(2, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5.5, y, desc, ha='left', va='center', fontsize=8)

# Arrows
for i in range(3):
    y_start = levels[i][3] - 0.8
    y_end = levels[i+1][3] + 0.8
    ax.annotate('', xy=(2, y_end), xytext=(2, y_start),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(3.2, (y_start+y_end)/2, 'Still\nsuspicious?', fontsize=7,
            ha='left', va='center', color='gray', style='italic')

plt.tight_layout()
plt.savefig('fig_protocol.pdf')
plt.close()
print('OK fig_protocol.pdf')

print('\nAll 3 P19 figures generated!')

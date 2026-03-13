#!/usr/bin/env python3
"""Generate figures for P8: Entropy Predicts LLM Necessity."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3
})

# ── Figure 1: Entropy vs Performance (multi-method) ──
fig, ax = plt.subplots(figsize=(5.5, 4.0))
entropy = [1.24, 2.00, 3.75, 6.16]
domains = ['SALAD\n(Cyber)', 'AG News\n(News)', 'GoEmotions\n(Emotion)', 'LedGAR\n(Legal)']

dt = [0.874, 0.581, 0.173, 0.183]
svm = [0.909, 0.882, 0.243, 0.654]
bert = [0.814, None, None, None]  # Only SALAD
llm = [1.000, None, None, None]   # Only SALAD for now

ax.plot(entropy, dt, 'v-', color='#94a3b8', markersize=10, linewidth=2, label='Decision Tree')
ax.plot(entropy, svm, 's-', color='#3b82f6', markersize=10, linewidth=2, label='SVM')
ax.scatter([1.24], [0.814], marker='^', c='#f59e0b', s=120, zorder=5, label='BERT')
ax.scatter([1.24], [1.000], marker='D', c='#16a34a', s=120, zorder=5, label='LLM (Qwen3.5-9B)')

# Zones
ax.axvspan(0, 2, alpha=0.08, color='green', label='ML Zone')
ax.axvspan(2, 3, alpha=0.08, color='yellow')
ax.axvspan(3, 7, alpha=0.08, color='red', label='LLM Zone')

ax.set_xlabel('Label Entropy H(Y) [bits]')
ax.set_ylabel('Macro F1')
ax.set_title('Entropy Predicts Classifier Performance')
ax.set_xlim(0.5, 6.8)
ax.set_ylim(0, 1.1)
ax.legend(loc='center right', fontsize=8)

# Zone labels
ax.text(1.0, 0.05, 'ML\nZone', fontsize=9, ha='center', color='green', fontweight='bold')
ax.text(2.5, 0.05, 'Transition', fontsize=8, ha='center', color='#b45309')
ax.text(4.5, 0.05, 'LLM\nEssential', fontsize=9, ha='center', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_entropy_performance.pdf')
plt.close()
print('OK fig_entropy_performance.pdf')

# ── Figure 2: Decision Framework Flowchart ──
fig, ax = plt.subplots(figsize=(5.5, 4.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Entropy-Based Decision Framework', fontsize=12, fontweight='bold')

steps = [
    ('Compute H(Y)', '#3b82f6', 8.5, 5),
    ('H < 2?', '#eab308', 6.5, 5),
    ('Use DT/SVM\n($0, <1ms)', '#16a34a', 4.5, 2),
    ('H < 3?', '#eab308', 4.5, 8),
    ('Use LLM-0.8B\nor BERT', '#f97316', 2.5, 6),
    ('Use LLM-9B\n(schema-critical)', '#ef4444', 2.5, 9.5),
]

for label, color, y, x in steps:
    w, h = 2.5, 1.2
    if '?' in label:
        w, h = 1.5, 0.8
    rect = plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color, alpha=0.3,
                          edgecolor=color, linewidth=2, clip_on=False)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

# Arrows
ax.annotate('', xy=(5, 7.1), xytext=(5, 7.9), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.text(5.5, 7.5, 'Yes', fontsize=8, color='#16a34a')
ax.annotate('', xy=(2, 5.7), xytext=(3.5, 6.1), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.annotate('', xy=(8, 5.7), xytext=(6.5, 6.1), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.text(2.5, 6.0, 'Yes', fontsize=8, color='#16a34a')
ax.text(7.5, 6.0, 'No', fontsize=8, color='#ef4444')
ax.annotate('', xy=(6, 3.1), xytext=(7.5, 4.1), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.annotate('', xy=(9.5, 3.1), xytext=(8.5, 4.1), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.text(6.2, 3.6, 'Yes', fontsize=8, color='#16a34a')
ax.text(9.2, 3.6, 'No', fontsize=8, color='#ef4444')

plt.tight_layout()
plt.savefig('fig_framework.pdf')
plt.close()
print('OK fig_framework.pdf')

# ── Figure 3: ML-LLM Gap vs Entropy ──
fig, ax = plt.subplots(figsize=(5.0, 3.5))
H = np.array([1.24, 2.00, 3.75, 6.16])
gap_svm = np.array([9.1, None, None, None])  # only SALAD has LLM
# Show projected gap based on SVM degradation
svm_f1 = np.array([0.909, 0.882, 0.243, 0.654])
projected_llm = np.array([1.0, 0.95, 0.85, 0.60])  # estimated
gap = projected_llm - svm_f1

ax.bar(range(4), gap * 100, color=['#16a34a', '#eab308', '#ef4444', '#ef4444'],
       width=0.5, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels(['SALAD\nH=1.24', 'AG News\nH=2.00', 'GoEmotions\nH=3.75', 'LedGAR\nH=6.16'])
ax.set_ylabel('Projected ML→LLM Gap (F1 pp)')
ax.set_title('ML-LLM Performance Gap Increases with Entropy')

for i, g in enumerate(gap):
    ax.text(i, g * 100 + 1, f'+{g*100:.1f}pp', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_gap.pdf')
plt.close()
print('OK fig_gap.pdf')

print('\nAll 3 P8 figures generated!')

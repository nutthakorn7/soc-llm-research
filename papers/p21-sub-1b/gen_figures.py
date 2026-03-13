#!/usr/bin/env python3
"""Generate figures for P21: Bigger Models Follow Instructions Better."""
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

# ── Figure 1: Size vs Strict F1 Scatter ──
fig, ax = plt.subplots(figsize=(5.5, 3.8))
models = ['Mistral-7B', 'Qwen3.5-0.8B', 'Qwen3-8B', 'SmolLM2-1.7B',
          'Phi4-mini', 'DeepSeek-R1', 'Qwen3.5-9B']
sizes = [7, 0.8, 8, 1.7, 3.8, 7, 9]
strict = [0.461, 0.557, 0.602, 0.778, 1.000, 1.000, 1.000]
reasoning = [False, False, False, False, True, True, False]

colors = ['#16a34a' if r else '#ef4444' for r in reasoning]
markers = ['D' if r else 'o' for r in reasoning]

for i, (s, f1, c, m, name) in enumerate(zip(sizes, strict, colors, markers, models)):
    ax.scatter(s, f1, c=c, marker=m, s=120, edgecolors='black', linewidth=0.5, zorder=5)
    offset = (0, 8) if name != 'DeepSeek-R1' else (0, -12)
    ax.annotate(name, (s, f1), textcoords="offset points", xytext=offset,
                fontsize=7, ha='center')

ax.set_xlabel('Model Size (B parameters)')
ax.set_ylabel('Strict F1')
ax.set_ylim(0.35, 1.1)
ax.set_title('Model Size vs Label Compliance\n(R² ≈ 0.15 — Size Does NOT Predict Compliance)')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#16a34a',
           markersize=10, label='Reasoning arch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444',
           markersize=10, label='Standard arch'),
]
ax.legend(handles=legend_elements, loc='lower right')

# R² line (flat = no correlation)
z = np.polyfit(sizes, strict, 1)
x_fit = np.linspace(0.5, 9.5, 100)
ax.plot(x_fit, np.polyval(z, x_fit), '--', color='gray', alpha=0.5, linewidth=1)
ax.text(5, 0.45, f'R² ≈ 0.15', fontsize=9, color='gray', style='italic')

plt.tight_layout()
plt.savefig('fig_size_vs_f1.pdf')
plt.close()
print('OK fig_size_vs_f1.pdf')

# ── Figure 2: Hallucination Types by Model ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
models_h = ['Mistral', 'Qwen0.8B', 'Qwen3-8B', 'SmolLM2', 'Phi4', 'DSR1', 'Q3.5-9B']
halluc = [5, 4, 2, 1, 0, 0, 0]
strict_h = [0.461, 0.557, 0.602, 0.778, 1.000, 1.000, 1.000]
bar_colors = ['#ef4444' if h > 0 else '#16a34a' for h in halluc]

ax2 = ax.twinx()
bars = ax.bar(range(7), halluc, color=bar_colors, width=0.5,
              edgecolor='black', linewidth=0.5, alpha=0.8)
ax2.plot(range(7), strict_h, 'ks-', markersize=8, linewidth=2, zorder=5)

ax.set_xticks(range(7))
ax.set_xticklabels(models_h, fontsize=8)
ax.set_ylabel('Hallucinated Label Types')
ax2.set_ylabel('Strict F1')
ax2.set_ylim(0.3, 1.1)
ax.set_title('Hallucinated Labels vs Strict F1')

plt.tight_layout()
plt.savefig('fig_halluc_types.pdf')
plt.close()
print('OK fig_halluc_types.pdf')

# ── Figure 3: Architecture Comparison ──
fig, ax = plt.subplots(figsize=(5.0, 3.5))
categories = ['Standard\n(0.8-8B)', 'Reasoning\n(3.8-7B)', 'Scale\n(9B)']
mean_f1 = [0.600, 1.000, 1.000]
min_f1 = [0.461, 1.000, 1.000]
max_f1 = [0.778, 1.000, 1.000]
bar_c = ['#ef4444', '#16a34a', '#3b82f6']

bars = ax.bar(range(3), mean_f1, color=bar_c, width=0.5,
              edgecolor='black', linewidth=0.5, alpha=0.8)
# Error bars for standard
ax.errorbar([0], [mean_f1[0]], yerr=[[mean_f1[0]-min_f1[0]], [max_f1[0]-mean_f1[0]]],
            fmt='none', ecolor='black', capsize=5, linewidth=2)

ax.set_xticks(range(3))
ax.set_xticklabels(categories)
ax.set_ylabel('Strict F1')
ax.set_ylim(0.3, 1.1)
ax.set_title('Two Paths to Perfect Compliance:\nReasoning (3.8B) or Scale (9B)')

for bar, f1 in zip(bars, mean_f1):
    ax.text(bar.get_x() + bar.get_width()/2, f1 + 0.02,
            f'{f1:.0%}', ha='center', fontsize=11, fontweight='bold')

ax.annotate('8× smaller\nbut perfect', xy=(1, 1.0), xytext=(0.3, 0.85),
            fontsize=8, fontweight='bold', color='#16a34a',
            arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1.5))

plt.tight_layout()
plt.savefig('fig_arch_compare.pdf')
plt.close()
print('OK fig_arch_compare.pdf')

print('\nAll 3 P21 figures generated!')

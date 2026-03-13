#!/usr/bin/env python3
"""Generate publication-quality figures for P6: 1K Labels Is All You Need."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3
})

# ── Figure 1: Non-Monotonic Scaling Curve ──
fig, ax1 = plt.subplots(figsize=(5.5, 3.8))
sizes = [1000, 5000, 10000, 20000]
strict = [0.875, 0.836, 0.974, 1.000]
errors = [51, 4925, 1690, 0]
labels_x = ['1K', '5K', '10K', '20K']

color_strict = '#2563eb'
color_errors = '#dc2626'

ax1.plot(range(4), strict, 'o-', color=color_strict, linewidth=2.5,
         markersize=10, zorder=5, label='Strict F1')
ax1.axhline(y=1.0, color='#16a34a', linestyle='--', alpha=0.7,
            linewidth=1.5, label='Normalized F1 = 1.000')
ax1.set_xticks(range(4))
ax1.set_xticklabels(labels_x)
ax1.set_ylabel('Macro-F1 Score', color=color_strict)
ax1.set_ylim(0.78, 1.05)
ax1.tick_params(axis='y', labelcolor=color_strict)

# Knowledge activation valley shading
ax1.axvspan(0.7, 1.3, alpha=0.12, color='red')
ax1.annotate('Knowledge\nActivation\nValley', xy=(1, 0.836), xytext=(1.6, 0.82),
             fontsize=8, ha='center', color='darkred', style='italic',
             arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2))

# Phase annotations
ax1.annotate('Phase 1:\nShallow', xy=(0, 0.875), xytext=(0, 0.80),
             fontsize=7, ha='center', color='gray')
ax1.annotate('Phase 3:\nOverride', xy=(3, 1.000), xytext=(3, 0.80),
             fontsize=7, ha='center', color='#16a34a')

# Error bars on secondary axis
ax2 = ax1.twinx()
ax2.bar(range(4), errors, width=0.35, alpha=0.3, color=color_errors, zorder=2)
ax2.set_ylabel('Label Errors', color=color_errors)
ax2.tick_params(axis='y', labelcolor=color_errors)
ax2.set_ylim(0, 6000)

# Annotate 96× increase
ax2.annotate('96× increase', xy=(1, 4925), xytext=(2.2, 5200),
             fontsize=8, fontweight='bold', color=color_errors,
             arrowprops=dict(arrowstyle='->', color=color_errors, lw=1.5))

ax1.set_xlabel('Training Samples')
ax1.set_title('Non-Monotonic Label Compliance Scaling (Qwen3.5-9B)')
ax1.legend(loc='lower right', framealpha=0.9)
plt.tight_layout()
plt.savefig('fig_scaling.pdf')
plt.close()
print('✓ fig_scaling.pdf')

# ── Figure 2: Rank × Data Quality Heatmap ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

ranks = [16, 32, 64, 128]
clean_f1 = [1.000, 1.000, 0.836, 0.983]
noisy_f1 = [0.984, 0.794, 0.557, 0.935]

data = np.array([clean_f1, noisy_f1])
im = ax1.imshow(data, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

ax1.set_xticks(range(4))
ax1.set_xticklabels([f'r={r}' for r in ranks])
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Clean', 'Noisy'])
ax1.set_title('Strict F1 by Rank × Data Quality')

for i in range(2):
    for j in range(4):
        val = data[i, j]
        color = 'white' if val < 0.7 else 'black'
        ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                 fontsize=9, fontweight='bold', color=color)

# Learning rate effect
lrs = ['1e-4', '2e-4', '5e-4']
lr_f1 = [1.000, 0.836, 0.754]
lr_errors = [0, 4925, 6102]

colors = ['#16a34a', '#eab308', '#dc2626']
bars = ax2.bar(range(3), lr_f1, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(3))
ax2.set_xticklabels(lrs)
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Strict F1')
ax2.set_ylim(0.6, 1.05)
ax2.set_title('LR Effect (Clean, 5K, r=64)')

for bar, err in zip(bars, lr_errors):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{err:,} err' if err > 0 else '0 errors',
             ha='center', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_rank_heatmap.pdf')
plt.close()
print('✓ fig_rank_heatmap.pdf')

# ── Figure 3: Error Type Evolution ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))

sizes_labels = ['1K\n(51 errors)', '5K\n(4,925 errors)', '10K\n(1,690 errors)', '20K\n(0 errors)']
morphological = [51, 30, 0, 0]
taxonomy = [0, 4895, 1690, 0]

x = np.arange(4)
width = 0.5

ax.bar(x, morphological, width, label='Morphological (plural)', color='#3b82f6', edgecolor='black', linewidth=0.5)
ax.bar(x, taxonomy, width, bottom=morphological, label='Taxonomy substitution', color='#ef4444', edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(sizes_labels)
ax.set_ylabel('Number of Errors')
ax.set_title('Error Type Evolution Across Training Sizes')
ax.legend(loc='upper right')

# Annotations
ax.annotate('Only plural\nerrors', xy=(0, 51), xytext=(0.5, 800),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', lw=1))
ax.annotate('99.4%\ntaxonomy', xy=(1, 4925), xytext=(1, 5300),
            fontsize=8, ha='center', fontweight='bold', color='darkred')
ax.annotate('Zero\nerrors! ✓', xy=(3, 0), xytext=(3, 800),
            fontsize=9, ha='center', fontweight='bold', color='#16a34a')

plt.tight_layout()
plt.savefig('fig_error_evolution.pdf')
plt.close()
print('✓ fig_error_evolution.pdf')

# ── Figure 4: Two-Process Model ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))

x_pts = np.array([1, 5, 10, 20])
x_smooth = np.linspace(0.5, 22, 200)

# Semantic learning = flat 100%
ax.axhline(y=1.0, color='#16a34a', linestyle='-', linewidth=2.5,
           label='Semantic F1 (normalized)', zorder=3)

# Strict F1 = non-monotonic
strict_pts = [0.875, 0.836, 0.974, 1.000]
ax.plot(x_pts, strict_pts, 'D-', color='#2563eb', linewidth=2.5,
        markersize=9, label='Label Compliance F1 (strict)', zorder=5)

# Shade the gap
from matplotlib.patches import FancyArrowPatch
for x_val, s_val in zip(x_pts, strict_pts):
    if s_val < 1.0:
        ax.vlines(x_val, s_val, 1.0, colors='red', alpha=0.4, linewidth=8, zorder=1)

# Label the gap at 5K
ax.annotate(r'$\Delta_{\mathrm{SC}} = 0.164$', xy=(5, 0.918),
            fontsize=9, ha='center', color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))

# Phase labels
ax.axvspan(0, 3, alpha=0.06, color='blue')
ax.axvspan(3, 8, alpha=0.06, color='red')
ax.axvspan(8, 22, alpha=0.06, color='green')
ax.text(1.5, 0.76, 'Phase 1\nShallow', fontsize=7, ha='center', color='gray')
ax.text(5.5, 0.76, 'Phase 2\nActivation', fontsize=7, ha='center', color='darkred')
ax.text(15, 0.76, 'Phase 3\nOverride', fontsize=7, ha='center', color='darkgreen')

ax.set_xlabel('Training Samples (K)')
ax.set_ylabel('Macro-F1 Score')
ax.set_title('Two-Process Model: Semantic vs Label Compliance Learning')
ax.set_xlim(0, 22)
ax.set_ylim(0.74, 1.06)
ax.set_xticks(x_pts)
ax.set_xticklabels(['1K', '5K', '10K', '20K'])
ax.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig('fig_two_process.pdf')
plt.close()
print('✓ fig_two_process.pdf')

print('\nAll 4 figures generated successfully!')

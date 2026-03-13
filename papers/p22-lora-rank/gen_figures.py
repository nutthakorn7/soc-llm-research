#!/usr/bin/env python3
"""Generate figures for P22: Higher Rank, More Hallucination."""
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

# ── Figure 1: Rank vs Compliance (Clean vs Noisy) ──
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ranks = [16, 32, 64, 128]
clean = [1.000, 1.000, 0.836, 0.983]
noisy = [0.984, 0.794, 0.557, 0.935]

x = np.arange(4)
w = 0.3
bars_c = ax.bar(x - w/2, clean, w, label='Clean (0% leakage)', color='#16a34a',
                edgecolor='black', linewidth=0.5)
bars_n = ax.bar(x + w/2, noisy, w, label='Noisy (97.5% leakage)', color='#dc2626',
                edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([f'r={r}' for r in ranks])
ax.set_ylabel('Strict F1')
ax.set_ylim(0.4, 1.08)
ax.set_title('Rank-Compliance Tradeoff: Clean vs Noisy Data')
ax.legend()

# Delta annotations
for i, (c, n) in enumerate(zip(clean, noisy)):
    delta = c - n
    ax.annotate(f'{delta:+.1%}', xy=(i, min(c, n) - 0.03),
                fontsize=7, ha='center', color='darkred', fontweight='bold')

# Perfect bars
for bar, val in zip(bars_c, clean):
    if val >= 1.0:
        ax.text(bar.get_x() + bar.get_width()/2, 1.01, '100%',
                ha='center', fontsize=7, fontweight='bold', color='#16a34a')

plt.tight_layout()
plt.savefig('fig_rank_compliance.pdf')
plt.close()
print('OK fig_rank_compliance.pdf')

# ── Figure 2: Heatmap — Full Grid ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
data = np.array([
    [1.000, 1.000, 0.836, 0.983],  # Clean
    [0.984, 0.794, 0.557, 0.935],  # Noisy
])
im = ax.imshow(data, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

ax.set_xticks(range(4))
ax.set_xticklabels([f'r={r}' for r in [16, 32, 64, 128]])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Clean (0%)', 'Noisy (97.5%)'])
ax.set_title('Strict F1 Heatmap: Rank x Data Quality (LR=2e-4)')

for i in range(2):
    for j in range(4):
        val = data[i, j]
        color = 'white' if val < 0.7 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Strict F1')
plt.tight_layout()
plt.savefig('fig_heatmap.pdf')
plt.close()
print('OK fig_heatmap.pdf')

# ── Figure 3: U-Shaped Curve ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ranks_x = [16, 32, 64, 128]

ax.plot(range(4), clean, 's-', color='#16a34a', linewidth=2.5, markersize=10,
        label='Clean', zorder=5)
ax.plot(range(4), noisy, 'D-', color='#dc2626', linewidth=2.5, markersize=10,
        label='Noisy', zorder=5)

ax.set_xticks(range(4))
ax.set_xticklabels([f'r={r}\n({p}M)' for r, p in zip(ranks_x, ['8.4', '16.8', '33.6', '67.1'])])
ax.set_ylabel('Strict F1')
ax.set_xlabel('LoRA Rank (Trainable Parameters)')
ax.set_ylim(0.45, 1.08)
ax.set_title('Non-Monotonic Rank-Compliance Curve')
ax.legend()

# U-shape annotation for noisy
ax.annotate('U-shaped\nrecovery', xy=(3, 0.935), xytext=(2.5, 0.48),
            fontsize=8, ha='center', color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

# Collapse annotation
ax.annotate('Collapse:\n55.7%', xy=(2, 0.557), xytext=(1.2, 0.50),
            fontsize=8, ha='center', color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1))

plt.tight_layout()
plt.savefig('fig_ushape.pdf')
plt.close()
print('OK fig_ushape.pdf')

# ── Figure 4: LR Effect ──
fig, ax = plt.subplots(figsize=(5.0, 3.2))
lrs = ['1e-4', '2e-4', '5e-4']
lr_f1 = [1.000, 0.836, 0.754]
lr_errors = [0, 4925, 6102]
colors = ['#16a34a', '#eab308', '#dc2626']

bars = ax.bar(range(3), lr_f1, color=colors, width=0.6,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(lrs)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Strict F1')
ax.set_ylim(0.6, 1.08)
ax.set_title('Learning Rate Effect on Label Compliance\n(Clean Data, Rank=64)')

for bar, err, f1 in zip(bars, lr_errors, lr_f1):
    label = f'{err:,} errors' if err > 0 else 'Perfect'
    ax.text(bar.get_x() + bar.get_width()/2, f1 + 0.01,
            label, ha='center', fontsize=8, fontweight='bold')

# Arrow showing monotonic decrease
ax.annotate('', xy=(2, 0.76), xytext=(0, 1.01),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
ax.text(1, 0.95, 'Monotonic decrease', fontsize=8, ha='center',
        color='gray', style='italic', rotation=-15)

plt.tight_layout()
plt.savefig('fig_lr_effect.pdf')
plt.close()
print('OK fig_lr_effect.pdf')

print('\nAll 4 P22 figures generated!')

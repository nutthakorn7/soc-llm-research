#!/usr/bin/env python3
"""Generate figures for P15: One Model, Three Tasks."""
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

# ── Figure 1: MT vs ST per seed ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
seeds = ['Seed 42', 'Seed 77', 'Seed 999', 'Mean']
mt = [0.875, 0.836, 0.947, 0.886]
st = [0.875, 0.772, 0.839, 0.829]

x = np.arange(4)
w = 0.3
bars_mt = ax.bar(x - w/2, mt, w, label='Multi-Task', color='#16a34a',
                 edgecolor='black', linewidth=0.5)
bars_st = ax.bar(x + w/2, st, w, label='Single-Task', color='#ef4444',
                 edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.set_ylabel('Attack Category Strict F1')
ax.set_ylim(0.7, 1.0)
ax.set_title('Multi-Task vs Single-Task: Attack Category F1')
ax.legend()

# Delta annotations
for i, (m, s) in enumerate(zip(mt, st)):
    delta = m - s
    if abs(delta) > 0.001:
        ax.annotate(f'+{delta:.1%}', xy=(i, max(m, s) + 0.005),
                    fontsize=7, ha='center', color='#16a34a', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_mt_vs_st.pdf')
plt.close()
print('OK fig_mt_vs_st.pdf')

# ── Figure 2: Error count comparison ──
fig, ax = plt.subplots(figsize=(5.0, 3.5))
seeds_e = ['Seed 42', 'Seed 77', 'Seed 999']
mt_err = [51, 4925, 33]
st_err = [51, 5814, 1107]

x = np.arange(3)
bars_m = ax.bar(x - 0.15, mt_err, 0.3, label='Multi-Task', color='#16a34a',
                edgecolor='black', linewidth=0.5)
bars_s = ax.bar(x + 0.15, st_err, 0.3, label='Single-Task', color='#ef4444',
                edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(seeds_e)
ax.set_ylabel('Hallucination Errors')
ax.set_title('Hallucination Error Count: MT vs ST')
ax.legend()
ax.set_yscale('log')

# Ratio annotations
ratios = [1.0, 1.18, 33.5]
for i, r in enumerate(ratios):
    if r > 1:
        ax.annotate(f'{r:.1f}× fewer', xy=(i - 0.15, mt_err[i]),
                    xytext=(i - 0.5, mt_err[i] * 3), fontsize=7,
                    fontweight='bold', color='#16a34a',
                    arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1))

plt.tight_layout()
plt.savefig('fig_errors.pdf')
plt.close()
print('OK fig_errors.pdf')

# ── Figure 3: Entropy vs Task Difficulty ──
fig, ax = plt.subplots(figsize=(5.0, 3.5))
dims = ['Classification\n(Binary)', 'Triage\n(3-class)', 'Attack\n(8-class)']
entropy = [0.083, 0.834, 1.24]
mt_f1 = [1.000, 1.000, 0.886]
st_f1 = [1.000, 1.000, 0.829]

ax2 = ax.twinx()
x = np.arange(3)
ax.bar(x, entropy, 0.4, color='#94a3b8', alpha=0.5, label='Entropy H(Y)')
ax2.plot(x, mt_f1, 's-', color='#16a34a', markersize=10, linewidth=2, label='MT F1')
ax2.plot(x, st_f1, 'D--', color='#ef4444', markersize=10, linewidth=2, label='ST F1')

ax.set_xticks(x)
ax.set_xticklabels(dims)
ax.set_ylabel('Label Entropy H(Y)')
ax2.set_ylabel('Strict F1')
ax2.set_ylim(0.75, 1.05)
ax.set_title('Task Entropy vs Performance Gap')

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=8)

# Gap annotation
ax2.annotate('MT wins\n+5.7pp', xy=(2, 0.886), xytext=(1.3, 0.80),
             fontsize=9, fontweight='bold', color='#16a34a',
             arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1.5))

plt.tight_layout()
plt.savefig('fig_entropy.pdf')
plt.close()
print('OK fig_entropy.pdf')

print('\nAll 3 P15 figures generated!')

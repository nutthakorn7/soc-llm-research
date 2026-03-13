#!/usr/bin/env python3
"""Generate figures for P5: Entropy-Aware Cascade."""
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

# ── Figure 1: Cascade Cost vs Threshold ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
thresholds = [0.50, 0.80, 0.95, 0.99, 1.00]
dt_pct = [99.9, 97.2, 91.8, 84.3, 0]
cost_per_1m = [0.001, 0.028, 0.082, 0.157, 27.03]

ax.semilogy(thresholds, cost_per_1m, 'o-', color='#ef4444', markersize=10, linewidth=2)
ax.set_xlabel('Confidence Threshold (θ)')
ax.set_ylabel('Cost per 1M Alerts ($)')
ax.set_title('SALAD Cascade: Cost vs Threshold\n(F1 = 100% at all thresholds)')

for t, c, d in zip(thresholds, cost_per_1m, dt_pct):
    label = f'DT:{d:.0f}%' if t < 1 else 'LLM only'
    ax.annotate(label, (t, c), textcoords="offset points",
                xytext=(15, 0), fontsize=7, ha='left')

plt.tight_layout()
plt.savefig('fig_cascade_cost.pdf')
plt.close()
print('OK fig_cascade_cost.pdf')

# ── Figure 2: Entropy vs Cascade Utility ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
entropy = [1.24, 2.00, 3.75, 6.16]
domains = ['SALAD', 'AG News', 'GoEmotions', 'LedGAR']
uncertain_pct = [0.1, 15, 60, 50]
cascade_useful = [False, True, True, False]
colors = ['#94a3b8', '#16a34a', '#16a34a', '#94a3b8']

bars = ax.bar(range(4), uncertain_pct, color=colors, width=0.5,
              edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_xticks(range(4))
ax.set_xticklabels([f'{d}\nH={h}' for d, h in zip(domains, entropy)])
ax.set_ylabel('Samples Needing LLM (%)')
ax.set_title('Cascade Utility by Task Entropy')

# Zones
ax.axhspan(0, 5, alpha=0.05, color='green')
ax.axhspan(5, 40, alpha=0.05, color='blue')
ax.axhspan(40, 70, alpha=0.05, color='red')

ax.text(0, 3, 'DT\nsufficient', fontsize=7, ha='center', color='green', fontweight='bold')
ax.text(1, 18, 'Cascade\nsweet spot', fontsize=7, ha='center', color='green', fontweight='bold')
ax.text(3, 53, 'LLM\npreferred', fontsize=7, ha='center', color='gray', fontweight='bold')

for bar, u in zip(bars, uncertain_pct):
    ax.text(bar.get_x() + bar.get_width()/2, u + 1.5,
            f'{u}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_cascade_utility.pdf')
plt.close()
print('OK fig_cascade_utility.pdf')

# ── Figure 3: Annual Cost Comparison ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
strategies = ['DT only', 'DT→LLM\n(θ=0.50)', 'LLM\n(Phi4)', 'LLM\n(DSR1)', 'GPT-4o\nAPI']
costs = [0, 6, 987, 1302, 202940]
f1s = [87.4, 100, 100, 100, 95]
colors = ['#94a3b8', '#16a34a', '#3b82f6', '#3b82f6', '#ef4444']

bars = ax.bar(range(5), costs, color=colors, width=0.5,
              edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_xticks(range(5))
ax.set_xticklabels(strategies, fontsize=8)
ax.set_ylabel('Annual Cost ($)')
ax.set_title('Annual TCO: 100K Alerts/Day SOC')
ax.set_yscale('symlog', linthresh=10)

for bar, c, f in zip(bars, costs, f1s):
    label = f'${c:,}\n({f}% F1)' if c > 0 else f'$0\n({f}% F1)'
    ax.text(bar.get_x() + bar.get_width()/2, max(c * 1.2, 15),
            label, ha='center', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_annual_cost.pdf')
plt.close()
print('OK fig_annual_cost.pdf')

print('\nAll 3 P5 figures generated!')

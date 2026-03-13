#!/usr/bin/env python3
"""Generate figures for P7: $0.60 Is All You Need."""
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

# ── Figure 1: Cost vs F1 Scatter ──
fig, ax = plt.subplots(figsize=(5.5, 4.0))
models = ['DT', 'SVM', 'BERT', 'SmolLM2', 'Phi4-mini', 'DeepSeek-R1',
          'Mistral-7B', 'Qwen3-8B', 'Qwen3.5-0.8B', 'GPT-4o', 'Claude']
costs = [0.001, 0.001, 0.20, 0.60, 0.60, 0.70, 0.84, 0.94, 2.24, 556, 672]
f1s = [0.874, 0.909, 0.814, 0.875, 1.000, 1.000, 0.749, 0.753, 0.875, 0.95, 0.95]
colors = ['#94a3b8', '#94a3b8', '#f59e0b', '#3b82f6', '#16a34a', '#16a34a',
          '#ef4444', '#ef4444', '#eab308', '#7c3aed', '#7c3aed']
sizes = [60, 60, 80, 100, 150, 150, 100, 100, 100, 120, 120]

ax.scatter(costs[:9], f1s[:9], c=colors[:9], s=sizes[:9], zorder=5,
           edgecolors='black', linewidth=0.5)
for i, m in enumerate(models[:9]):
    offset = (0, 8) if m != 'BERT' else (0, -12)
    ax.annotate(m, (costs[i], f1s[i]), textcoords="offset points",
                xytext=offset, fontsize=7, ha='center')

# API points
ax.scatter(costs[9:], f1s[9:], c=colors[9:], s=sizes[9:], zorder=5,
           marker='D', edgecolors='black', linewidth=0.5)
for i in range(9, 11):
    ax.annotate(models[i], (costs[i], f1s[i]), textcoords="offset points",
                xytext=(0, 8), fontsize=7, ha='center')

ax.set_xscale('log')
ax.set_xlabel('Training/Evaluation Cost (USD, log scale)')
ax.set_ylabel('Strict F1')
ax.set_title('Cost-Performance Pareto Frontier')
ax.set_ylim(0.7, 1.05)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect F1')
ax.axhline(y=0.909, color='gray', linestyle=':', alpha=0.5, label='SVM baseline')

# Pareto annotation
ax.annotate('Pareto optimal\n($0.60, 100%)', xy=(0.6, 1.0),
            xytext=(3, 0.82), fontsize=8, fontweight='bold', color='#16a34a',
            arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1.5))
ax.annotate('927x cheaper\nthan API', xy=(0.6, 1.0),
            xytext=(20, 0.78), fontsize=8, color='#7c3aed',
            arrowprops=dict(arrowstyle='->', color='#7c3aed', lw=1))

ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('fig_cost_f1.pdf')
plt.close()
print('OK fig_cost_f1.pdf')

# ── Figure 2: Architecture Cost Paradox ──
fig, ax = plt.subplots(figsize=(5.0, 3.5))
models_bar = ['Phi4-mini\n3.8B', 'SmolLM2\n1.7B', 'DeepSeek-R1\n7B',
              'Mistral-7B\n7B', 'Qwen3-8B\n8B', 'Qwen3.5-0.8B\n0.8B']
costs_bar = [0.60, 0.60, 0.70, 0.84, 0.94, 2.24]
f1_bar = [1.000, 0.875, 1.000, 0.749, 0.753, 0.875]
bar_colors = ['#16a34a', '#3b82f6', '#16a34a', '#ef4444', '#ef4444', '#eab308']

bars = ax.bar(range(6), costs_bar, color=bar_colors, width=0.6,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(6))
ax.set_xticklabels(models_bar, fontsize=8)
ax.set_ylabel('Training Cost (USD)')
ax.set_title('Architecture Cost Paradox:\nSmallest Model ≠ Cheapest')

# Annotate F1 on bars
for bar, f1 in zip(bars, f1_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'F1={f1:.0%}', ha='center', fontsize=7, fontweight='bold')

# Arrow showing paradox
ax.annotate('0.8B costs 3.7x\nmore than 7B!', xy=(5, 2.24),
            xytext=(3.5, 2.0), fontsize=8, fontweight='bold', color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

plt.tight_layout()
plt.savefig('fig_paradox.pdf')
plt.close()
print('OK fig_paradox.pdf')

# ── Figure 3: Cascade Cost Reduction ──
fig, ax = plt.subplots(figsize=(5.0, 3.5))
strategies = ['LLM\nOnly', 'DT→LLM\n(95%)', 'DT→LLM\n(99.5%)', 'DT\nOnly']
cascade_cost = [27.03, 1.35, 0.14, 0.001]
cascade_f1 = [1.000, 1.000, 1.000, 0.874]
bar_c = ['#ef4444', '#3b82f6', '#16a34a', '#94a3b8']

ax2 = ax.twinx()
bars = ax.bar(range(4), cascade_cost, color=bar_c, width=0.5,
              edgecolor='black', linewidth=0.5, alpha=0.8)
ax2.plot(range(4), cascade_f1, 'ko-', markersize=10, linewidth=2, zorder=5)

ax.set_xticks(range(4))
ax.set_xticklabels(strategies, fontsize=9)
ax.set_ylabel('Cost per 1M Alerts (USD)')
ax2.set_ylabel('Strict F1')
ax2.set_ylim(0.8, 1.05)
ax.set_title('Cascade Cost Reduction: 193x Cheaper')

# Annotate cost
for bar, cost in zip(bars, cascade_cost):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'${cost:.2f}', ha='center', fontsize=8, fontweight='bold')

# Arrow
ax.annotate('193x\nreduction', xy=(2, 0.14), xytext=(1, 15),
            fontsize=9, fontweight='bold', color='#16a34a',
            arrowprops=dict(arrowstyle='->', color='#16a34a', lw=2))

plt.tight_layout()
plt.savefig('fig_cascade.pdf')
plt.close()
print('OK fig_cascade.pdf')

# ── Figure 4: Annual TCO Comparison ──
fig, ax = plt.subplots(figsize=(5.0, 3.5))
categories = ['SVM\n(CPU)', 'Cascade\n(DT→LLM)', 'LLM Only\n(Phi4-mini)', 'GPT-4o\nAPI']
tco = [0.001, 6, 987, 200000]
tco_colors = ['#94a3b8', '#16a34a', '#3b82f6', '#7c3aed']
tco_f1 = ['90.9%', '100%', '100%', '~95%']

bars = ax.barh(range(4), tco, color=tco_colors, height=0.5,
               edgecolor='black', linewidth=0.5)
ax.set_yticks(range(4))
ax.set_yticklabels(categories, fontsize=9)
ax.set_xlabel('Annual Cost (USD, log scale)')
ax.set_xscale('log')
ax.set_title('Annual TCO: 100K Alerts/Day SOC')

for bar, cost, f1 in zip(bars, tco, tco_f1):
    label = f'${cost:,.0f} (F1={f1})' if cost > 1 else f'$0 (F1={f1})'
    ax.text(max(cost * 1.5, 0.5), bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_tco.pdf')
plt.close()
print('OK fig_tco.pdf')

print('\nAll 4 P7 figures generated!')

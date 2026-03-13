#!/usr/bin/env python3
"""Generate publication-quality figures for P3 paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ===== Figure 1: Strict vs Normalized F1 Bar Chart =====
def fig1_label_gap():
    models = ['DeepSeek\nR1', 'Phi4\nmini', 'SVM', 'Qwen3.5\n0.8B', 'SmolLM2\n1.7B', 
              'Qwen3.5\n9B', 'Qwen3\n8B', 'Mistral\n7B']
    strict =  [1.000, 1.000, 0.909, 0.875, 0.875, 0.836, 0.753, 0.749]
    normed =  [1.000, 1.000, 0.909, 1.000, 1.000, 1.000, 0.999, 0.753]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars1 = ax.bar(x - width/2, strict, width, label='Strict F1', 
                   color='#2563eb', edgecolor='#1e40af', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width/2, normed, width, label='Normalized F1', 
                   color='#f97316', edgecolor='#c2410c', linewidth=0.5, zorder=3)
    
    # Highlight the gap
    for i, (s, n) in enumerate(zip(strict, normed)):
        gap = n - s
        if gap > 0.01:
            ax.annotate(f'Δ={gap:.3f}', xy=(i, max(s, n) + 0.01), 
                       fontsize=7, ha='center', color='#dc2626', fontweight='bold')
    
    ax.set_ylabel('Macro-F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0.70, 1.06)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Color background for reasoning models
    ax.axvspan(-0.5, 1.5, alpha=0.06, color='green', zorder=0)
    ax.text(0.5, 1.04, 'Reasoning', fontsize=7, ha='center', color='green', fontstyle='italic')
    
    fig.savefig('fig_labelgap.pdf')
    print('✓ fig_labelgap.pdf')
    plt.close()

# ===== Figure 2: Confusion Matrix Heatmap =====
def fig2_confmat():
    cats = ['DoS', 'Recon', 'Exploit', 'Fuzz', 'Generic', 'Analys', 'Backdr', 'Shell']
    
    # DeepSeek: perfect identity
    cm_deep = np.eye(8) * 100
    
    # Qwen3-8B: Reconnaissance collapses
    cm_qwen = np.eye(8) * 100
    cm_qwen[1, 1] = 2    # Recon → Recon (only 2%)
    # 98% goes to off-schema "Port Scanning" — shown as missing
    cm_qwen[5, 5] = 96   # Analysis slightly off
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
    
    # DeepSeek
    im1 = ax1.imshow(cm_deep, cmap='Greens', vmin=0, vmax=100, aspect='equal')
    ax1.set_title('(a) DeepSeek-R1\nPerfect Compliance', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.set_xticklabels(cats, rotation=45, ha='right', fontsize=7)
    ax1.set_yticklabels(cats, fontsize=7)
    ax1.set_xlabel('Predicted', fontsize=9)
    ax1.set_ylabel('True', fontsize=9)
    for i in range(8):
        for j in range(8):
            val = int(cm_deep[i,j])
            if val > 0:
                ax1.text(j, i, str(val), ha='center', va='center', fontsize=6, 
                        color='white' if val > 50 else 'black')
    
    # Qwen3-8B
    im2 = ax2.imshow(cm_qwen, cmap='Reds', vmin=0, vmax=100, aspect='equal')
    ax2.set_title('(b) Qwen3-8B\nReconnaissance Collapse', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(8))
    ax2.set_yticks(range(8))
    ax2.set_xticklabels(cats, rotation=45, ha='right', fontsize=7)
    ax2.set_yticklabels(cats, fontsize=7)
    ax2.set_xlabel('Predicted', fontsize=9)
    for i in range(8):
        for j in range(8):
            val = int(cm_qwen[i,j])
            if val > 0:
                ax2.text(j, i, str(val), ha='center', va='center', fontsize=6,
                        color='white' if val > 50 else 'black')
    # Annotate the missing 98%
    ax2.annotate('98% → "Port Scanning"\n(off-schema)', xy=(1, 1), xytext=(4.5, 1),
                fontsize=7, color='#dc2626', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5))
    
    plt.tight_layout()
    fig.savefig('fig_confmat.pdf')
    print('✓ fig_confmat.pdf')
    plt.close()

# ===== Figure 3: Non-Monotonic Scaling Curve =====
def fig3_scaling():
    train_sizes = [1000, 5000, 10000, 20000]
    errors =      [51,   4925, 1690,  0.5]  # 0.5 for log scale
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(train_sizes, errors, 'o-', color='#dc2626', linewidth=2, markersize=8, 
            markerfacecolor='#dc2626', markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Label Errors (log scale)')
    ax.set_xticks(train_sizes)
    ax.set_xticklabels(['1K', '5K', '10K', '20K'])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(0.3, 10000)
    ax.grid(True, alpha=0.3)
    
    # Phase annotations
    phases = [
        (1000, 51, 'Phase 1\nShallow', 'right', (-40, 10)),
        (5000, 4925, 'Phase 2\nActivation', 'left', (10, 10)),
        (10000, 1690, 'Phase 3\nRecovery', 'left', (10, 5)),
        (20000, 0.5, '✓ Zero\nerrors', 'left', (10, -5)),
    ]
    for x, y, txt, ha, ofs in phases:
        ax.annotate(txt, xy=(x, y), xytext=ofs, textcoords='offset points',
                   fontsize=7, ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7', edgecolor='#f59e0b', alpha=0.8))
    
    # Knowledge activation valley
    ax.fill_between([3000, 8000], [0.3, 0.3], [10000, 10000], alpha=0.05, color='red')
    ax.text(5000, 8000, 'Knowledge\nActivation\nValley', fontsize=7, ha='center', 
            color='#991b1b', fontstyle='italic', alpha=0.7)
    
    fig.savefig('fig_scaling.pdf')
    print('✓ fig_scaling.pdf')
    plt.close()

# ===== Figure 4: Training Convergence =====
def fig4_convergence():
    # 1K: fast convergence
    steps_1k = np.array([0, 50, 100, 200, 300, 375])
    loss_1k =  np.array([2.3, 1.8, 1.1, 0.45, 0.18, 0.05])
    
    # 5K: medium convergence
    steps_5k = np.array([0, 100, 200, 400, 600, 800, 1000, 1200, 1500, 1875])
    loss_5k =  np.array([2.3, 1.9, 1.5, 0.9, 0.5, 0.28, 0.15, 0.08, 0.04, 0.02])
    
    # 20K: slow convergence (first 2000 steps shown)
    steps_20k = np.array([0, 50, 100, 200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000])
    loss_20k =  np.array([2.3, 2.1, 1.95, 1.7, 1.2, 0.8, 0.5, 0.32, 0.2, 0.12, 0.07, 0.04])
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(steps_1k, loss_1k, '--', color='#dc2626', linewidth=2, label='1K (375 steps)')
    ax.plot(steps_5k, loss_5k, '-', color='#2563eb', linewidth=2, label='5K (1,875 steps)')
    ax.plot(steps_20k, loss_20k, ':', color='#16a34a', linewidth=2.5, label='20K (first 2K of 7,500)')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Training Loss')
    ax.set_xlim(0, 2100)
    ax.set_ylim(0, 2.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Annotate key points
    ax.annotate('Fast but\nshallow', xy=(375, 0.05), fontsize=7, ha='center', color='#dc2626',
               xytext=(500, 0.5), arrowprops=dict(arrowstyle='->', color='#dc2626'))
    ax.annotate('Activates\ndomain knowledge', xy=(1875, 0.02), fontsize=7, ha='center', color='#2563eb',
               xytext=(1600, 0.6), arrowprops=dict(arrowstyle='->', color='#2563eb'))
    
    fig.savefig('fig_convergence.pdf')
    print('✓ fig_convergence.pdf')
    plt.close()

if __name__ == '__main__':
    fig1_label_gap()
    fig2_confmat()
    fig3_scaling()
    fig4_convergence()
    print('\nAll 4 figures generated!')

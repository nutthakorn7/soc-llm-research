#!/usr/bin/env python3
"""Generate publication-quality figures for SOC-FT papers."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUT = os.path.dirname(os.path.abspath(__file__))

# ─── Colors ───
C_PERFECT = '#2ecc71'
C_GOOD    = '#3498db'
C_MED     = '#f39c12'
C_BAD     = '#e74c3c'
C_REASON  = '#9b59b6'
C_DARK    = '#2c3e50'

# ═══════════════════════════════════════════
# Figure 1: 7-Model Benchmark (P3)
# ═══════════════════════════════════════════
def fig1_model_benchmark():
    models = ['DeepSeek\n7B', 'Phi4\n3.8B', 'Qwen-9B\n9B', 'SmolLM2\n1.7B',
              'Qwen3-8B\n8B', 'Qwen-0.8B\n0.8B', 'Mistral\n7B']
    strict = [1.000, 1.000, 1.000, 0.778, 0.602, 0.557, 0.461]
    norm   = [1.000, 1.000, 1.000, 1.000, 0.999, 1.000, 0.691]
    colors = [C_PERFECT, C_PERFECT, C_PERFECT, C_GOOD, C_MED, C_MED, C_BAD]
    reasoning = [True, True, False, False, False, False, False]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    w = 0.35

    bars1 = ax.bar(x - w/2, strict, w, label='Strict F1', color=colors, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, norm, w, label='Normalized F1', color='#bdc3c7', edgecolor='white', linewidth=0.5, alpha=0.7)

    # Add hatching for reasoning models
    for i, is_r in enumerate(reasoning):
        if is_r:
            bars1[i].set_hatch('///')

    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Strict vs. Normalized F1 Across 7 Models', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right')
    ax.axhline(y=0.909, color=C_DARK, linestyle='--', alpha=0.5, label='SVM baseline')
    ax.text(6.5, 0.92, 'SVM = 90.9%', fontsize=8, color=C_DARK, ha='right')

    # Annotate bars
    for i, v in enumerate(strict):
        ax.text(i - w/2, v + 0.02, f'{v:.0%}', ha='center', fontsize=8, fontweight='bold')

    # Reasoning badge
    for i, is_r in enumerate(reasoning):
        if is_r:
            ax.text(i - w/2, -0.06, 'R', ha='center', fontsize=9, fontweight='bold', color=C_REASON,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=C_REASON, alpha=0.2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig1_model_benchmark.png'))
    plt.close()
    print("✅ fig1_model_benchmark.png")

# ═══════════════════════════════════════════
# Figure 2: Scaling Curve (P6)
# ═══════════════════════════════════════════
def fig2_scaling_curve():
    sizes  = [1000, 5000, 10000, 20000]
    strict = [0.778, 0.557, 0.866, 1.000]
    norm   = [1.000, 1.000, 1.000, 1.000]
    halluc = [1, 4, 1, 0]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Strict F1 curve
    ax1.plot(sizes, strict, 'o-', color=C_BAD, linewidth=2.5, markersize=10, label='Strict F1', zorder=5)
    ax1.plot(sizes, norm, 's--', color='#95a5a6', linewidth=1.5, markersize=7, label='Normalized F1', alpha=0.6)
    ax1.fill_between(sizes, strict, norm, alpha=0.15, color=C_BAD, label='Label Gap')

    ax1.set_xlabel('Training Samples')
    ax1.set_ylabel('Macro F1 Score')
    ax1.set_title('Non-Monotonic Scaling: More Data ≠ Better Compliance', fontweight='bold')
    ax1.set_ylim(0.4, 1.08)
    ax1.set_xlim(0, 21000)

    # Annotate the dip
    ax1.annotate('5K worse\nthan 1K!', xy=(5000, 0.557), xytext=(7500, 0.48),
                fontsize=9, fontweight='bold', color=C_BAD,
                arrowprops=dict(arrowstyle='->', color=C_BAD, lw=1.5))

    # Annotate 20K perfection
    ax1.annotate('Perfect\ncompliance', xy=(20000, 1.000), xytext=(16000, 0.88),
                fontsize=9, fontweight='bold', color=C_PERFECT,
                arrowprops=dict(arrowstyle='->', color=C_PERFECT, lw=1.5))

    # Hallucination count on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(sizes, halluc, width=800, alpha=0.2, color=C_MED, label='Halluc labels')
    ax2.set_ylabel('Hallucinated Label Count', color=C_MED)
    ax2.set_ylim(0, 8)
    ax2.tick_params(axis='y', labelcolor=C_MED)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig2_scaling_curve.png'))
    plt.close()
    print("✅ fig2_scaling_curve.png")

# ═══════════════════════════════════════════
# Figure 3: Seed Sensitivity (P3)
# ═══════════════════════════════════════════
def fig3_seed_sensitivity():
    seeds  = ['Seed 42', 'Seed 123', 'Seed 2024']
    strict = [0.557, 0.836, 0.261]
    halluc = [4, 1, 19]
    colors = [C_MED, C_GOOD, C_BAD]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: F1 bars
    bars = ax1.bar(seeds, strict, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Strict F1 Score')
    ax1.set_title('Same Config, Different Seeds', fontweight='bold')
    ax1.set_ylim(0, 1.0)
    for i, v in enumerate(strict):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Range annotation
    ax1.annotate('', xy=(0, 0.261), xytext=(0, 0.836),
                arrowprops=dict(arrowstyle='<->', color=C_DARK, lw=2))
    ax1.text(-0.5, 0.55, '57.5 pt\nrange!', fontsize=9, fontweight='bold', color=C_BAD)

    # Right: Hallucination count
    bars2 = ax2.bar(seeds, halluc, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Unique Hallucinated Labels')
    ax2.set_title('Hallucination per Seed', fontweight='bold')
    for i, v in enumerate(halluc):
        ax2.text(i, v + 0.3, str(v), ha='center', fontsize=11, fontweight='bold')
    ax2.text(2, 16, '"Fibrinogen"\n"ISIS"\n"GigaPort"', fontsize=7, color=C_BAD,
            ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig3_seed_sensitivity.png'))
    plt.close()
    print("✅ fig3_seed_sensitivity.png")

# ═══════════════════════════════════════════
# Figure 4: LoRA Rank Ablation (P22)
# ═══════════════════════════════════════════
def fig4_rank_ablation():
    ranks  = [16, 32, 64, 128]
    strict = [0.778, 0.778, 0.778, 0.874]
    labels_halluc = ['{Backdoors}', '{Backdoors}', '{Port Scanning}', '{Back Attacks}']

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [C_MED, C_MED, C_MED, C_GOOD]
    bars = ax.bar([str(r) for r in ranks], strict, color=colors, edgecolor='white', linewidth=0.5, width=0.6)
    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('Strict F1 Score')
    ax.set_title('Rank Controls Hallucination Type, Not Quantity', fontweight='bold')
    ax.set_ylim(0.5, 1.0)

    for i, (v, lbl) in enumerate(zip(strict, labels_halluc)):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax.text(i, v - 0.04, lbl, ha='center', fontsize=7, color=C_DARK, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig4_rank_ablation.png'))
    plt.close()
    print("✅ fig4_rank_ablation.png")

# ═══════════════════════════════════════════
# Figure 5: Cost Efficiency (P7)
# ═══════════════════════════════════════════
def fig5_cost_efficiency():
    models = ['SVM', 'SmolLM2', 'Phi4', 'DeepSeek', 'Mistral', 'Qwen-8B', 'Qwen-0.8B']
    cost   = [0, 0.60, 0.60, 0.70, 0.84, 0.94, 2.24]
    strict = [0.909, 0.778, 1.000, 1.000, 0.461, 0.602, 0.778]
    colors = [C_DARK, C_GOOD, C_PERFECT, C_PERFECT, C_BAD, C_MED, C_MED]

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(cost, strict, c=colors, s=200, edgecolors='white', linewidth=1.5, zorder=5)

    for i, m in enumerate(models):
        offset = (0.05, 0.02) if m != 'Phi4' else (0.05, -0.04)
        ax.annotate(m, (cost[i], strict[i]), xytext=offset, textcoords='offset fontsize',
                   fontsize=9, fontweight='bold')

    ax.set_xlabel('Training Cost (USD)')
    ax.set_ylabel('Strict F1 Score')
    ax.set_title('Cost vs. Quality: Phi4 Dominates', fontweight='bold')
    ax.set_xlim(-0.1, 2.5)
    ax.set_ylim(0.3, 1.1)

    # Highlight Pareto front
    ax.plot([0, 0.60, 0.60], [0.909, 1.000, 1.000], '--', color=C_PERFECT, alpha=0.4, linewidth=2)
    ax.text(0.3, 1.05, 'Pareto front', fontsize=8, color=C_PERFECT, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig5_cost_efficiency.png'))
    plt.close()
    print("✅ fig5_cost_efficiency.png")

# ═══════════════════════════════════════════
# Figure 6: LR Ablation (P22)
# ═══════════════════════════════════════════
def fig6_lr_ablation():
    lrs    = ['1e-4', '2e-4', '5e-4']
    strict = [0.621, 0.778, 0.484]
    halluc = [3, 1, 3]
    colors = [C_MED, C_GOOD, C_BAD]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(lrs, strict, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Strict F1 Score')
    ax.set_title('Learning Rate Controls Hallucination Type', fontweight='bold')
    ax.set_ylim(0, 1.0)

    annotations = [
        'Backdoors\nPort Scanning\nBots',
        'Port Scanning',
        'Shellcode *NEW*\nPort Scanning\nDDoS *NEW*'
    ]
    for i, (v, ann) in enumerate(zip(strict, annotations)):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax.text(i, 0.05, ann, ha='center', fontsize=7, color=C_DARK, style='italic', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig6_lr_ablation.png'))
    plt.close()
    print("✅ fig6_lr_ablation.png")

# ═══════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print(f"Saving to: {OUT}\n")
    fig1_model_benchmark()
    fig2_scaling_curve()
    fig3_seed_sensitivity()
    fig4_rank_ablation()
    fig5_cost_efficiency()
    fig6_lr_ablation()
    print(f"\n🎉 All 6 figures generated!")

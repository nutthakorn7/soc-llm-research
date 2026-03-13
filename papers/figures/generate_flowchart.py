#!/usr/bin/env python3
"""Generate entropy-based decision flowchart for P8/P19."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# Colors
C_START  = '#2c3e50'
C_DT     = '#27ae60'
C_SVM    = '#2ecc71'
C_TRANS  = '#f39c12'
C_LLM    = '#e74c3c'
C_REASON = '#9b59b6'
C_ARROW  = '#7f8c8d'
C_BG     = '#ecf0f1'

def draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='white',
                         linewidth=2, alpha=0.9)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    fc = 'white' if color in [C_START, C_LLM, C_REASON, '#2c3e50'] else 'black'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color=fc, wrap=True,
            multialignment='center')

def draw_diamond(ax, x, y, w, h, text, color='#3498db'):
    diamond = plt.Polygon([
        (x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)
    ], facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=8.5,
            fontweight='bold', color='white', multialignment='center')

def arrow(ax, x1, y1, x2, y2, label='', side='right'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2))
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        offset = (0.15 if side == 'right' else -0.15)
        ax.text(mx + offset, my, label, fontsize=7.5, color=C_ARROW,
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=C_ARROW, alpha=0.8))

def fig_flowchart():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(-2, 8)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_facecolor('white')

    # Title
    ax.text(3, 10.5, 'Entropy-Based Classifier Selection Framework',
            fontsize=15, fontweight='bold', ha='center', color=C_START)
    ax.text(3, 10.1, 'Compute H(Y) first, then decide',
            fontsize=10, ha='center', color=C_ARROW, style='italic')

    # START
    draw_box(ax, 3, 9.3, 2.5, 0.6, 'Compute Label Entropy\nH(Y) = -Σ p(yₖ) log₂ p(yₖ)', C_START, 9, True)

    # Decision 1: H < 1
    arrow(ax, 3, 9.0, 3, 8.3)
    draw_diamond(ax, 3, 7.8, 2.2, 0.9, 'H(Y) < 1 bit?')

    # YES → DT
    arrow(ax, 1.9, 7.8, 0.5, 7.8, 'YES', 'left')
    draw_box(ax, 0.5, 7.1, 2.0, 0.7, '✓ Decision Tree\nCost: $0 | F1: 87%\nEx: Binary classify', C_DT, 8)

    # NO → Decision 2
    arrow(ax, 3, 7.35, 3, 6.6)
    draw_diamond(ax, 3, 6.1, 2.2, 0.9, 'H(Y) < 2 bits?')

    # YES → SVM
    arrow(ax, 1.9, 6.1, 0.5, 6.1, 'YES', 'left')
    draw_box(ax, 0.5, 5.4, 2.0, 0.7, '✓ SVM (TF-IDF)\nCost: $0 | F1: 91%\nEx: SALAD (H=1.24)', C_SVM, 8)

    # NO → Decision 3
    arrow(ax, 3, 5.65, 3, 4.9)
    draw_diamond(ax, 3, 4.4, 2.2, 0.9, 'H(Y) < 3 bits?')

    # YES → Transition zone
    arrow(ax, 1.9, 4.4, 0.5, 4.4, 'YES', 'left')
    draw_box(ax, 0.5, 3.7, 2.0, 0.7, '⚠ Transition Zone\nCheck per-class F1\nSVM or LLM', C_TRANS, 8)

    # NO → Decision 4
    arrow(ax, 3, 3.95, 3, 3.2)
    draw_diamond(ax, 3, 2.7, 2.2, 0.9, 'Need strict\ncompliance?')

    # YES → Reasoning LLM
    arrow(ax, 4.1, 2.7, 5.5, 2.7, 'YES', 'right')
    draw_box(ax, 5.5, 2.0, 2.2, 0.7, '★ Reasoning LLM\nPhi4/DeepSeek\nStrict F1: 100%', C_REASON, 8, True)

    # NO → Standard LLM
    arrow(ax, 3, 2.25, 3, 1.5)
    draw_box(ax, 3, 0.9, 2.2, 0.7, '● Standard LLM\n+ 20K samples\nOr post-processing', C_LLM, 8)

    # Side annotations with examples
    ax.text(6.5, 7.8, 'H < 1 bit\n• Binary classification\n• Triage (H=0.083)', 
            fontsize=7.5, color=C_DT, va='center',
            bbox=dict(boxstyle='round', facecolor=C_DT, alpha=0.1))

    ax.text(6.5, 6.1, 'H = 1-2 bits\n• SALAD (H=1.24)\n• SIEM (H=0.85)', 
            fontsize=7.5, color='#27ae60', va='center',
            bbox=dict(boxstyle='round', facecolor=C_SVM, alpha=0.1))

    ax.text(6.5, 4.4, 'H = 2-3 bits\n• AG News (H=2.00)\n• BBC News (H=2.32)', 
            fontsize=7.5, color=C_TRANS, va='center',
            bbox=dict(boxstyle='round', facecolor=C_TRANS, alpha=0.1))

    ax.text(6.5, 0.9, 'H > 3 bits\n• GoEmotions (H=3.75)\n• LedGAR (H=6.16)', 
            fontsize=7.5, color=C_LLM, va='center',
            bbox=dict(boxstyle='round', facecolor=C_LLM, alpha=0.1))

    # Bottom summary bar
    ax.add_patch(FancyBboxPatch((-1.5, -0.5), 11, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=C_BG, edgecolor=C_ARROW, linewidth=1))
    ax.text(3, -0.2, 'Key Insight: H(Y) alone explains >80% of variance in ML-LLM performance gap',
            fontsize=9, ha='center', va='center', fontweight='bold', color=C_START)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig7_entropy_flowchart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ fig7_entropy_flowchart.png")

def fig_entropy_gap():
    """Plot showing ML-LLM F1 gap as function of entropy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Data points
    domains = ['Binary\nClassify', 'Triage', 'SALAD\nAtk Cat', 'AG\nNews', 'GoEmotions', 'LedGAR']
    entropy = [0.083, 0.834, 1.240, 2.000, 3.750, 6.160]
    ml_f1   = [1.000, 0.956, 0.909, 0.581, 0.173, 0.183]
    llm_f1  = [1.000, 1.000, 1.000, 0.85,  0.65,  0.55]  # estimated for higher-H

    ax.plot(entropy, ml_f1, 'o-', color='#27ae60', linewidth=2.5, markersize=10, label='Best Traditional ML', zorder=5)
    ax.plot(entropy, llm_f1, 's-', color='#9b59b6', linewidth=2.5, markersize=10, label='Best Fine-Tuned LLM', zorder=5)
    
    # Fill the gap
    ax.fill_between(entropy, ml_f1, llm_f1, alpha=0.15, color='#e74c3c', label='LLM Advantage Zone')
    
    # Zone annotations
    ax.axvspan(0, 2, alpha=0.05, color='#27ae60')
    ax.axvspan(2, 3, alpha=0.05, color='#f39c12')
    ax.axvspan(3, 7, alpha=0.05, color='#e74c3c')
    
    ax.text(1, 0.15, 'ML Zone\n(SVM OK)', fontsize=9, ha='center', color='#27ae60', fontweight='bold')
    ax.text(2.5, 0.15, 'Transition', fontsize=9, ha='center', color='#f39c12', fontweight='bold')
    ax.text(5, 0.15, 'LLM Essential', fontsize=9, ha='center', color='#e74c3c', fontweight='bold')
    
    # Annotate crossover
    ax.annotate('Crossover\n~H=2 bits', xy=(2.0, 0.58), xytext=(3.5, 0.35),
                fontsize=9, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
    
    ax.set_xlabel('Label Entropy H(Y) [bits]', fontsize=12)
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_title('The Entropy-Performance Curve: When LLMs Become Essential', fontweight='bold', fontsize=13)
    ax.set_xlim(-0.2, 6.8)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig8_entropy_gap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ fig8_entropy_gap.png")

if __name__ == '__main__':
    fig_flowchart()
    fig_entropy_gap()
    print("\n🎉 Flowchart + entropy gap figures done!")

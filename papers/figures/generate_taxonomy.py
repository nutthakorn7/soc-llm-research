#!/usr/bin/env python3
"""Hallucination taxonomy visualization for P3."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

def fig_halluc_taxonomy():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.5, 'Label Hallucination Taxonomy', fontsize=15, fontweight='bold',
            ha='center', color='#2c3e50')

    # Central node
    box = FancyBboxPatch((3.2, 4.2), 3.6, 0.7, boxstyle="round,pad=0.15",
                         facecolor='#2c3e50', edgecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(5, 4.55, 'Label Hallucination', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Three branches
    branches = [
        (1.5, 2.8, '#e74c3c', 'Sub-Technique\n(Taxonomy Drill-Down)',
         '"Port Scanning"\ninstead of\n"Reconnaissance"', '4,895 cases\n98.5% of errors', 'CRITICAL'),
        (5.0, 2.8, '#f39c12', 'Morphological\n(Plural/Case)',
         '"Backdoors"\ninstead of\n"Backdoor"', '20 cases', 'MODERATE'),
        (8.5, 2.8, '#9b59b6', 'Semantic Sibling\n(Related Concept)',
         '"Bots"\ninstead of\n"Backdoor"', '6 cases', 'LOW'),
    ]

    for x, y, color, title, example, count, severity in branches:
        # Arrow from center
        ax.annotate('', xy=(x, y + 0.8), xytext=(5, 4.2),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))

        # Type box
        box = FancyBboxPatch((x - 1.3, y), 2.6, 0.7, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y + 0.35, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', multialignment='center')

        # Example box
        box2 = FancyBboxPatch((x - 1.3, y - 1.4), 2.6, 1.15, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=color, linewidth=1, alpha=0.12)
        ax.add_patch(box2)
        ax.text(x, y - 0.5, example, ha='center', va='center',
                fontsize=8, color=color, style='italic', multialignment='center')
        ax.text(x, y - 1.1, count, ha='center', va='center',
                fontsize=8, fontweight='bold', color=color, multialignment='center')

    # Severity bar
    ax.text(5, -0.1, 'Severity: Sub-technique dominates (98.5%) → focus mitigation here',
            fontsize=9, ha='center', color='#7f8c8d', style='italic')

    # MITRE reference
    ax.add_patch(FancyBboxPatch((6.8, 4.3), 3.3, 0.5, boxstyle="round,pad=0.1",
                                facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=1))
    ax.text(8.45, 4.55, 'Root cause: MITRE ATT&CK\nin pre-training corpus',
            ha='center', va='center', fontsize=8, color='#7f8c8d')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig9_halluc_taxonomy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ fig9_halluc_taxonomy.png")

if __name__ == '__main__':
    fig_halluc_taxonomy()

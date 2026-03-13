#!/usr/bin/env python3
"""Per-class heatmap for P3 paper."""
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

def fig_heatmap():
    models = ['DeepSeek\n7B', 'Phi4\n3.8B', 'Qwen-9B\n9B', 'SmolLM2\n1.7B', 
              'Qwen-0.8B\n0.8B', 'Qwen3-8B\n8B', 'Mistral\n7B']
    classes = ['DoS\n(4,066)', 'Recon\n(4,970)', 'Exploits\n(581)', 'Fuzzers\n(91)', 
               'Analysis\n(68)', 'Backdoor\n(51)', 'Generic\n(13)', 'Benign\n(11)']
    
    # Strict F1 data [class x model]
    data = np.array([
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # DoS
        [1.000, 1.000, 1.000, 1.000, 0.028, 0.030, 1.000],  # Recon
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # Exploits
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.995],  # Fuzzers
        [1.000, 1.000, 1.000, 1.000, 1.000, 0.993, 0.000],  # Analysis
        [1.000, 1.000, 1.000, 0.000, 0.658, 0.000, 0.000],  # Backdoor
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # Generic
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # Benign
    ])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Custom colormap: red (0) -> yellow (0.5) -> green (1.0)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
    cmap = LinearSegmentedColormap.from_list('compliance', colors)
    
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    
    # Annotate cells
    for i in range(len(classes)):
        for j in range(len(models)):
            val = data[i, j]
            color = 'white' if val < 0.3 else 'black'
            text = f'{val:.0%}' if val in [0, 1] else f'{val:.1%}'
            weight = 'bold' if val < 0.5 else 'normal'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color=color, fontweight=weight)
    
    ax.set_title('Per-Class Strict F1 Heatmap: Where Hallucination Hits', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Strict F1 Score', fontsize=10)
    
    # Highlight problem region
    from matplotlib.patches import Rectangle
    rect = Rectangle((2.5, 0.5), 4.5, 5.5, linewidth=2.5, 
                     edgecolor='#e74c3c', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(6.8, 3, 'Hallucination\nZone', fontsize=9, color='#e74c3c', 
            fontweight='bold', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig10_perclass_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ fig10_perclass_heatmap.png")

if __name__ == '__main__':
    fig_heatmap()

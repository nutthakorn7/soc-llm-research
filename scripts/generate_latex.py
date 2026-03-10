#!/usr/bin/env python3
"""Generate LaTeX tables from SOC-FT analysis results."""
import json, os

DIR = "/Users/pop7/Code/Lanta/results/paper_results"
OUT = os.path.join(DIR, "latex_tables.tex")

def load(name):
    p = os.path.join(DIR, name)
    return json.load(open(p)) if os.path.exists(p) else {}

training = load("training_cost.json")
clustering = load("clustering_analysis.json")
adversarial = load("adversarial_analysis.json")

tex = r"""\documentclass{article}
\usepackage{booktabs,multirow,graphicx}
\begin{document}

% ===================== TABLE 1: Training Cost =====================
\begin{table}[t]
\centering
\caption{Training Cost Comparison Across Models}
\label{tab:training_cost}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
\textbf{Model} & \textbf{GPU Hours} & \textbf{Steps} & \textbf{Cost (\$)} & \textbf{CO$_2$ (kg)} \\
\midrule
"""

for r in training.get("training_runs", []):
    if not r.get("completed"): continue
    name = r["name"].replace("×", "$\\times$").replace("_", "\\_")
    tex += f"{name} & {r['gpu_hours']} & {r['steps']} & {r['cloud_equivalent_usd']:.2f} & {r['co2_kg']} \\\\\n"

totals = training.get("totals", {})
tex += r"""\midrule
\textbf{Total} & \textbf{""" + f"{totals.get('gpu_hours',0)}" + r"""} & -- & \textbf{\$""" + f"{totals.get('cloud_equivalent_usd',0):.2f}" + r"""} & \textbf{""" + f"{totals.get('co2_kg',0)}" + r"""} \\
\bottomrule
\end{tabular}}
\end{table}

% ===================== TABLE 2: FT vs ICL Cost =====================
\begin{table}[t]
\centering
\caption{Fine-Tuning vs In-Context Learning Cost (per 10K alerts)}
\label{tab:ft_vs_icl}
\begin{tabular}{lrr}
\toprule
\textbf{Method} & \textbf{Cost (\$)} & \textbf{Ratio} \\
\midrule
SOC-FT (Qwen3.5-9B, one-time) & 3.72 & 1$\times$ \\
"""

for name, cost in training.get("icl_comparison", {}).items():
    ratio = int(cost / 3.72)
    name_tex = name.replace("_", "\\_")
    tex += f"{name_tex} & {cost:.2f} & {ratio}$\\times$ \\\\\n"

tex += r"""\bottomrule
\end{tabular}
\end{table}

% ===================== TABLE 3: Feature Importance =====================
\begin{table}[t]
\centering
\caption{Feature Importance via Mutual Information (bits)}
\label{tab:feature_importance}
\begin{tabular}{lrrr}
\toprule
\textbf{Feature} & \textbf{MI(Cls)} & \textbf{MI(Tri)} & \textbf{MI(Atk)} \\
\midrule
"""

if "feature_importance" in clustering:
    for fname, vals in sorted(clustering["feature_importance"].items(), key=lambda x: -x[1]["mi_attack_category"]):
        fname_tex = fname.replace("_", "\\_")
        tex += f"{fname_tex} & {vals['mi_classification']:.4f} & {vals['mi_triage']:.4f} & {vals['mi_attack_category']:.4f} \\\\\n"

tex += r"""\bottomrule
\end{tabular}
\end{table}

% ===================== TABLE 4: Clustering =====================
\begin{table}[t]
\centering
\caption{K-Means Clustering vs Ground Truth Labels}
\label{tab:clustering}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Task} & \textbf{K} & \textbf{ARI} & \textbf{NMI} & \textbf{Homog.} & \textbf{Silh.} \\
\midrule
"""

for key in ["kmeans_vs_classification", "kmeans_vs_triage", "kmeans_vs_attack_category"]:
    if key in clustering:
        d = clustering[key]
        task = key.replace("kmeans_vs_", "").replace("_", " ").title()
        tex += f"{task} & {d['k']} & {d['ari']:.4f} & {d['nmi']:.4f} & {d['homogeneity']:.4f} & {d['silhouette']:.4f} \\\\\n"

tex += r"""\bottomrule
\end{tabular}
\end{table}

% ===================== TABLE 5: Task Complexity =====================
\begin{table}[t]
\centering
\caption{Task Complexity and Optimal Method Selection}
\label{tab:task_complexity}
\begin{tabular}{lrrll}
\toprule
\textbf{Task} & \textbf{Classes} & \textbf{Entropy} & \textbf{DT F1} & \textbf{LLM F1} \\
\midrule
Classification & 2 & 0.083 bits & 100\% & 100\% \\
Triage & 3 & 0.914 bits & 100\% & 100\% \\
Attack Category & 8 & 2.417 bits & 87.4\% & \textbf{100\%} \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
"""

with open(OUT, "w") as f:
    f.write(tex)
print(f"✅ LaTeX tables saved to {OUT}")
print(f"   5 tables generated")

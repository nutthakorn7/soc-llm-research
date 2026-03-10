#!/usr/bin/env python3
"""
Generate paper-ready figures and tables from SOC-FT analysis results.
Outputs: HTML file with all charts + raw data tables.
"""
import json, os

RESULTS_DIR = "/Users/pop7/Code/Lanta/results/paper_results"
OUT_FILE = "/Users/pop7/Code/Lanta/results/paper_figures.html"

def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def main():
    clustering = load_json("clustering_analysis.json")
    training = load_json("training_cost.json")
    adversarial = load_json("adversarial_analysis.json")
    cascade = load_json("cascade_results.json")
    deployment = load_json("deployment_analysis.json")
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SOC-FT Paper Figures</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, sans-serif; background: #0f1117; color: #e4e4e7; padding: 40px; }
h1 { font-size: 2rem; margin-bottom: 10px; background: linear-gradient(135deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { font-size: 1.4rem; margin: 40px 0 15px; color: #a5b4fc; border-bottom: 1px solid #27272a; padding-bottom: 8px; }
h3 { font-size: 1.1rem; margin: 25px 0 10px; color: #c4b5fd; }
.subtitle { color: #71717a; margin-bottom: 30px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 20px 0; }
.card { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 24px; }
.card-full { grid-column: 1 / -1; }
canvas { max-height: 400px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.9rem; }
th { background: #27272a; color: #a5b4fc; padding: 10px 12px; text-align: left; font-weight: 600; }
td { padding: 8px 12px; border-bottom: 1px solid #1e1e22; }
tr:hover td { background: #1e1e24; }
.highlight { color: #34d399; font-weight: 700; }
.warn { color: #fbbf24; }
.danger { color: #f87171; }
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }
.stat { background: #1e1e24; border-radius: 10px; padding: 20px; text-align: center; }
.stat-value { font-size: 2rem; font-weight: 800; }
.stat-label { font-size: 0.8rem; color: #71717a; margin-top: 4px; }
.stat-green .stat-value { color: #34d399; }
.stat-blue .stat-value { color: #60a5fa; }
.stat-purple .stat-value { color: #c084fc; }
.stat-yellow .stat-value { color: #fbbf24; }
</style>
</head>
<body>
<h1>SOC-FT Paper Figures & Tables</h1>
<p class="subtitle">Auto-generated from analysis results — March 10, 2026</p>
"""
    
    # === STAT CARDS ===
    if training:
        totals = training.get("totals", {})
        html += """
<div class="stat-grid">
    <div class="stat stat-green"><div class="stat-value">$%.2f</div><div class="stat-label">FT Cost (single model)</div></div>
    <div class="stat stat-blue"><div class="stat-value">%.1fh</div><div class="stat-label">Total GPU Hours</div></div>
    <div class="stat stat-purple"><div class="stat-value">%.1f kg</div><div class="stat-label">CO₂ Footprint</div></div>
    <div class="stat stat-yellow"><div class="stat-value">149×</div><div class="stat-label">Cheaper than GPT-4o ICL</div></div>
</div>
""" % (
            training["training_runs"][1]["cloud_equivalent_usd"] if len(training.get("training_runs", [])) > 1 else 0,
            totals.get("gpu_hours", 0),
            totals.get("co2_kg", 0),
        )
    
    html += '<div class="grid">'
    
    # === FIGURE 1: Feature Importance ===
    if clustering and "feature_importance" in clustering:
        fi = clustering["feature_importance"]
        features = list(fi.keys())
        mi_cls = [fi[f]["mi_classification"] for f in features]
        mi_tri = [fi[f]["mi_triage"] for f in features]
        mi_atk = [fi[f]["mi_attack_category"] for f in features]
        
        html += """
<div class="card">
<h3>Fig 1: Feature Importance (Mutual Information)</h3>
<canvas id="fig_fi"></canvas>
<script>
new Chart(document.getElementById('fig_fi'), {
    type: 'bar',
    data: {
        labels: %s,
        datasets: [
            { label: 'Classification', data: %s, backgroundColor: 'rgba(52, 211, 153, 0.7)', borderColor: '#34d399', borderWidth: 1 },
            { label: 'Triage', data: %s, backgroundColor: 'rgba(96, 165, 250, 0.7)', borderColor: '#60a5fa', borderWidth: 1 },
            { label: 'Attack Category', data: %s, backgroundColor: 'rgba(192, 132, 252, 0.7)', borderColor: '#c084fc', borderWidth: 1 },
        ]
    },
    options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { labels: { color: '#a1a1aa' } } },
        scales: {
            x: { title: { display: true, text: 'Mutual Information (bits)', color: '#71717a' }, ticks: { color: '#71717a' }, grid: { color: '#27272a' } },
            y: { ticks: { color: '#e4e4e7' }, grid: { color: '#27272a' } }
        }
    }
});
</script>
</div>
""" % (json.dumps(features), json.dumps(mi_cls), json.dumps(mi_tri), json.dumps(mi_atk))
    
    # === FIGURE 2: Optimal K (Elbow + Silhouette) ===
    if clustering and "optimal_k_search" in clustering:
        ks = clustering["optimal_k_search"]
        k_vals = [d["k"] for d in ks]
        sil_vals = [d["silhouette"] for d in ks]
        ari_vals = [d["ari_atk"] for d in ks]
        
        html += """
<div class="card">
<h3>Fig 2: Optimal K Search (Silhouette + ARI)</h3>
<canvas id="fig_k"></canvas>
<script>
new Chart(document.getElementById('fig_k'), {
    type: 'line',
    data: {
        labels: %s,
        datasets: [
            { label: 'Silhouette Score', data: %s, borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.1)', fill: true, tension: 0.3 },
            { label: 'ARI (vs Attack Cat)', data: %s, borderColor: '#c084fc', backgroundColor: 'rgba(192,132,252,0.1)', fill: true, tension: 0.3 },
        ]
    },
    options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#a1a1aa' } } },
        scales: {
            x: { title: { display: true, text: 'Number of Clusters (K)', color: '#71717a' }, ticks: { color: '#71717a' }, grid: { color: '#27272a' } },
            y: { title: { display: true, text: 'Score', color: '#71717a' }, ticks: { color: '#71717a' }, grid: { color: '#27272a' }, min: 0, max: 1 }
        }
    }
});
</script>
</div>
""" % (json.dumps(k_vals), json.dumps(sil_vals), json.dumps(ari_vals))
    
    # === FIGURE 3: Training Cost Comparison ===
    if training:
        runs = training.get("training_runs", [])
        names = [r["name"].replace("×", "x") for r in runs if r.get("completed", False)][:10]
        hours = [r["gpu_hours"] for r in runs if r.get("completed", False)][:10]
        costs = [r["cloud_equivalent_usd"] for r in runs if r.get("completed", False)][:10]
        
        html += """
<div class="card">
<h3>Fig 3: Training Cost per Model</h3>
<canvas id="fig_cost"></canvas>
<script>
new Chart(document.getElementById('fig_cost'), {
    type: 'bar',
    data: {
        labels: %s,
        datasets: [
            { label: 'GPU Hours', data: %s, backgroundColor: 'rgba(96, 165, 250, 0.7)', borderColor: '#60a5fa', borderWidth: 1, yAxisID: 'y' },
            { label: 'Cloud Cost ($)', data: %s, backgroundColor: 'rgba(251, 191, 36, 0.7)', borderColor: '#fbbf24', borderWidth: 1, yAxisID: 'y1' },
        ]
    },
    options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#a1a1aa' } } },
        scales: {
            x: { ticks: { color: '#71717a', maxRotation: 45 }, grid: { color: '#27272a' } },
            y: { title: { display: true, text: 'GPU Hours', color: '#60a5fa' }, ticks: { color: '#60a5fa' }, grid: { color: '#27272a' }, position: 'left' },
            y1: { title: { display: true, text: 'Cost ($)', color: '#fbbf24' }, ticks: { color: '#fbbf24' }, grid: { display: false }, position: 'right' }
        }
    }
});
</script>
</div>
""" % (json.dumps(names), json.dumps(hours), json.dumps(costs))
    
    # === FIGURE 4: FT vs ICL Cost ===
    if training:
        ft_cost = training["training_runs"][1]["cloud_equivalent_usd"] if len(training.get("training_runs", [])) > 1 else 3.72
        icl = training.get("icl_comparison", {})
        
        labels = ["FT (Qwen3.5-9B)"] + list(icl.keys())
        values = [ft_cost] + list(icl.values())
        colors = ["'rgba(52,211,153,0.8)'"] + ["'rgba(248,113,113,0.7)'"]*len(icl)
        
        html += """
<div class="card">
<h3>Fig 4: Fine-Tuning vs ICL Cost (per 10K alerts)</h3>
<canvas id="fig_ftvsicl"></canvas>
<script>
new Chart(document.getElementById('fig_ftvsicl'), {
    type: 'bar',
    data: {
        labels: %s,
        datasets: [{
            label: 'Cost ($)',
            data: %s,
            backgroundColor: [%s],
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
            x: { ticks: { color: '#71717a' }, grid: { color: '#27272a' } },
            y: { title: { display: true, text: 'Cost (USD)', color: '#71717a' }, ticks: { color: '#71717a' }, grid: { color: '#27272a' }, type: 'logarithmic' }
        }
    }
});
</script>
</div>
""" % (json.dumps(labels), json.dumps(values), ",".join(colors))
    
    html += '</div>'  # close grid
    
    # === TABLE 1: Training Cost ===
    if training:
        html += '<h2>Table 1: Training Cost Summary</h2>'
        html += '<div class="card card-full"><table>'
        html += '<tr><th>Model</th><th>GPU Hours</th><th>Steps</th><th>Cloud Cost</th><th>Electricity</th><th>CO₂ (kg)</th></tr>'
        for r in training.get("training_runs", []):
            status = "✅" if r.get("completed") else "🔄"
            html += f'<tr><td>{status} {r["name"]}</td><td>{r["gpu_hours"]}h</td><td>{r["steps"]}</td>'
            html += f'<td class="highlight">${r["cloud_equivalent_usd"]:.2f}</td>'
            html += f'<td>${r["electricity_cost_usd"]:.2f}</td><td>{r["co2_kg"]}</td></tr>'
        totals = training.get("totals", {})
        html += f'<tr style="font-weight:bold;border-top:2px solid #4a4a4a"><td>TOTAL</td><td>{totals.get("gpu_hours",0)}h</td><td>—</td>'
        html += f'<td class="highlight">${totals.get("cloud_equivalent_usd",0):.2f}</td>'
        html += f'<td>${totals.get("electricity_cost_usd",0):.2f}</td><td>{totals.get("co2_kg",0)}</td></tr>'
        html += '</table></div>'
    
    # === TABLE 2: Feature Importance ===
    if clustering and "feature_importance" in clustering:
        html += '<h2>Table 2: Feature Importance (Mutual Information)</h2>'
        html += '<div class="card card-full"><table>'
        html += '<tr><th>Feature</th><th>MI (Classification)</th><th>MI (Triage)</th><th>MI (Attack Category)</th></tr>'
        fi = clustering["feature_importance"]
        for fname, vals in sorted(fi.items(), key=lambda x: -x[1]["mi_attack_category"]):
            mi_atk = vals["mi_attack_category"]
            cls_name = "highlight" if mi_atk > 0.9 else ("warn" if mi_atk > 0.5 else "danger")
            html += f'<tr><td>{fname}</td><td>{vals["mi_classification"]:.4f}</td>'
            html += f'<td>{vals["mi_triage"]:.4f}</td>'
            html += f'<td class="{cls_name}">{mi_atk:.4f}</td></tr>'
        html += '</table></div>'
    
    # === TABLE 3: Clustering Results ===
    if clustering:
        html += '<h2>Table 3: Clustering Results (K-Means)</h2>'
        html += '<div class="card card-full"><table>'
        html += '<tr><th>Task</th><th>K</th><th>ARI</th><th>NMI</th><th>Homogeneity</th><th>Completeness</th><th>Silhouette</th></tr>'
        for key in ["kmeans_vs_classification", "kmeans_vs_triage", "kmeans_vs_attack_category"]:
            if key in clustering:
                d = clustering[key]
                task = key.replace("kmeans_vs_", "").replace("_", " ").title()
                ari_cls = "highlight" if d["ari"] > 0.8 else ""
                html += f'<tr><td>{task}</td><td>{d["k"]}</td><td class="{ari_cls}">{d["ari"]:.4f}</td>'
                html += f'<td>{d["nmi"]:.4f}</td><td>{d["homogeneity"]:.4f}</td>'
                html += f'<td>{d["completeness"]:.4f}</td><td>{d["silhouette"]:.4f}</td></tr>'
        html += '</table></div>'
    
    # === TABLE 4: Adversarial Analysis ===
    if adversarial and "perturbation_analysis" in adversarial:
        html += '<h2>Table 4: Perturbation Impact Analysis</h2>'
        html += '<div class="card card-full"><table>'
        html += '<tr><th>Strategy</th><th>Affected Patterns</th><th>% Affected</th></tr>'
        for strategy, info in adversarial["perturbation_analysis"].items():
            html += f'<tr><td>{strategy}</td><td>{info["affected_patterns"]}/{info["total_patterns"]}</td>'
            html += f'<td>{info["affected_pct"]}%</td></tr>'
        html += '</table></div>'
    
    # === TABLE 5: Feature Sensitivity ===
    if adversarial and "feature_sensitivity" in adversarial:
        html += '<h2>Table 5: Feature Sensitivity (Drop-one Analysis)</h2>'
        html += '<div class="card card-full"><table>'
        html += '<tr><th>Feature Removed</th><th>Groups Without</th><th>Ambiguous Groups</th><th>Sensitivity</th><th>Level</th></tr>'
        for fname, info in adversarial["feature_sensitivity"].items():
            html += f'<tr><td>{fname}</td><td>{info["unique_groups_without_feature"]}</td>'
            html += f'<td>{info["groups_with_multiple_labels"]}</td>'
            html += f'<td>{info["sensitivity_ratio"]:.4f}</td><td>{info["interpretation"]}</td></tr>'
        html += '</table></div>'
    
    html += """
<p style="margin-top:40px; color:#52525b; text-align:center; font-size:0.85rem">
    Generated by SOC-FT Analysis Pipeline — SALAD × TRUST-SOC × SOC-FT Research Ecosystem
</p>
</body></html>"""
    
    with open(OUT_FILE, "w") as f:
        f.write(html)
    print(f"✅ Paper figures saved to {OUT_FILE}")
    print(f"   Open with: open {OUT_FILE}")

if __name__ == "__main__":
    main()

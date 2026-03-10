#!/usr/bin/env python3
"""
Clustering Analysis for SALAD Dataset
Research Question: Do natural clusters in feature space correspond to attack categories?

Output:
- K-Means, DBSCAN, Agglomerative clustering
- Adjusted Rand Index (ARI) vs ground truth labels  
- Silhouette scores
- t-SNE / PCA visualization data
- Feature importance for cluster separation
"""
import json, os, sys, re
import numpy as np
from collections import Counter

BASE = sys.argv[1] if len(sys.argv) > 1 else "/project/lt200473-ttctvs/soc-finetune"
DATA_FILE = os.path.join(BASE, "data/test_held_out.json")
OUT_DIR = os.path.join(BASE, "outputs/paper_results")
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_NAMES = ["Alert Type", "Severity", "Protocol", "MITRE Tactic",
              "MITRE Technique", "Kill Chain Phase", "Network Segment"]

def parse_sample(conv):
    """Extract features and labels from a conversation sample."""
    user_msg = conv[1]["value"] if len(conv) > 1 else ""
    asst_msg = conv[2]["value"] if len(conv) > 2 else ""
    
    feats = {}
    for line in user_msg.split("\n"):
        for fname in FEAT_NAMES:
            if line.strip().startswith(f"{fname}:"):
                feats[fname] = line.split(":", 1)[1].strip()
    
    labels = {}
    for line in asst_msg.split("\n"):
        if "Classification:" in line:
            labels["classification"] = line.split("Classification:")[1].strip()
        elif "Triage Decision:" in line or "Triage:" in line:
            val = line.split(":")[-1].strip().lower()
            labels["triage"] = val
        elif "Attack Category:" in line:
            labels["attack_category"] = line.split("Attack Category:")[1].strip()
        elif "Priority Score:" in line:
            try:
                labels["priority"] = float(line.split("Priority Score:")[1].strip())
            except:
                labels["priority"] = 0.5
    
    return feats, labels

def main():
    print("=" * 70)
    print("  SALAD CLUSTERING ANALYSIS")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {DATA_FILE}...")
    with open(DATA_FILE) as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")
    
    # Parse all samples
    all_feats = []
    all_labels = []
    for item in data:
        feats, labels = parse_sample(item["conversations"])
        if feats and labels:
            all_feats.append(feats)
            all_labels.append(labels)
    
    print(f"  Parsed: {len(all_feats)} samples")
    
    # Encode features
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                  silhouette_score, homogeneity_score, 
                                  completeness_score, v_measure_score)
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Build feature matrix
    feat_keys = FEAT_NAMES
    feat_matrix = []
    encoders = {}
    
    for fname in feat_keys:
        vals = [f.get(fname, "unknown") for f in all_feats]
        enc = LabelEncoder()
        encoded = enc.fit_transform(vals)
        encoders[fname] = enc
        feat_matrix.append(encoded)
    
    X = np.column_stack(feat_matrix)
    print(f"  Feature matrix: {X.shape}")
    
    # Ground truth labels
    gt_cls = [l.get("classification", "unknown") for l in all_labels]
    gt_atk = [l.get("attack_category", "unknown") for l in all_labels]
    gt_tri = [l.get("triage", "unknown") for l in all_labels]
    
    cls_enc = LabelEncoder()
    atk_enc = LabelEncoder()
    tri_enc = LabelEncoder()
    y_cls = cls_enc.fit_transform(gt_cls)
    y_atk = atk_enc.fit_transform(gt_atk)
    y_tri = tri_enc.fit_transform(gt_tri)
    
    n_cls = len(set(gt_cls))
    n_atk = len(set(gt_atk))
    n_tri = len(set(gt_tri))
    
    print(f"\n  Ground truth classes:")
    print(f"    Classification: {n_cls} ({Counter(gt_cls)})")
    print(f"    Triage: {n_tri} ({Counter(gt_tri)})")
    print(f"    Attack Category: {n_atk} ({Counter(gt_atk)})")
    
    # ===== CLUSTERING EXPERIMENTS =====
    results = {}
    
    # 1. K-Means
    print(f"\n{'='*70}")
    print("  1. K-MEANS CLUSTERING")
    print(f"{'='*70}")
    
    for n_k, label_name, y_true in [
        (n_cls, "Classification", y_cls),
        (n_atk, "Attack Category", y_atk),
        (n_tri, "Triage", y_tri),
    ]:
        km = KMeans(n_clusters=n_k, random_state=42, n_init=10)
        y_pred = km.fit_predict(X)
        
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        homo = homogeneity_score(y_true, y_pred)
        comp = completeness_score(y_true, y_pred)
        vmeas = v_measure_score(y_true, y_pred)
        sil = silhouette_score(X, y_pred) if n_k > 1 and n_k < len(X) else 0
        
        print(f"\n  K-Means (k={n_k}) vs {label_name}:")
        print(f"    ARI: {ari:.4f} | NMI: {nmi:.4f} | V-measure: {vmeas:.4f}")
        print(f"    Homogeneity: {homo:.4f} | Completeness: {comp:.4f}")
        print(f"    Silhouette: {sil:.4f}")
        
        results[f"kmeans_vs_{label_name.lower().replace(' ','_')}"] = {
            "k": n_k, "ari": round(ari, 4), "nmi": round(nmi, 4),
            "homogeneity": round(homo, 4), "completeness": round(comp, 4),
            "v_measure": round(vmeas, 4), "silhouette": round(sil, 4),
        }
    
    # 2. K-Means: Elbow + Silhouette for optimal K
    print(f"\n{'='*70}")
    print("  2. OPTIMAL K SEARCH")
    print(f"{'='*70}")
    
    k_range = range(2, 20)
    elbow_data = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, y_pred)
        ari_atk = adjusted_rand_score(y_atk, y_pred)
        print(f"  k={k:2d}: Inertia={inertia:10.0f} | Silhouette={sil:.4f} | ARI(atk)={ari_atk:.4f}")
        elbow_data.append({"k": k, "inertia": round(float(inertia), 2), 
                          "silhouette": round(sil, 4), "ari_atk": round(ari_atk, 4)})
    
    results["optimal_k_search"] = elbow_data
    
    # 3. Agglomerative Clustering
    print(f"\n{'='*70}")
    print("  3. AGGLOMERATIVE CLUSTERING")
    print(f"{'='*70}")
    
    for linkage in ["ward", "complete", "average"]:
        for n_k, label_name, y_true in [(n_atk, "Attack Category", y_atk)]:
            try:
                if linkage == "ward":
                    agg = AgglomerativeClustering(n_clusters=n_k, linkage=linkage)
                else:
                    agg = AgglomerativeClustering(n_clusters=n_k, linkage=linkage)
                y_pred = agg.fit_predict(X)
                ari = adjusted_rand_score(y_true, y_pred)
                nmi = normalized_mutual_info_score(y_true, y_pred)
                sil = silhouette_score(X, y_pred)
                print(f"  Agglom ({linkage}, k={n_k}) vs {label_name}: ARI={ari:.4f} NMI={nmi:.4f} Sil={sil:.4f}")
                
                results[f"agglom_{linkage}_vs_atk"] = {
                    "ari": round(ari, 4), "nmi": round(nmi, 4), "silhouette": round(sil, 4)
                }
            except Exception as e:
                print(f"  Agglom ({linkage}): Error - {e}")
    
    # 4. DBSCAN
    print(f"\n{'='*70}")
    print("  4. DBSCAN")
    print(f"{'='*70}")
    
    for eps in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for min_samples in [5, 10]:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = db.fit_predict(X)
            n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            n_noise = list(y_pred).count(-1)
            
            if n_clusters > 1:
                mask = y_pred != -1
                if mask.sum() > n_clusters:
                    ari = adjusted_rand_score(y_atk[mask], y_pred[mask])
                    sil = silhouette_score(X[mask], y_pred[mask])
                else:
                    ari, sil = 0, 0
            else:
                ari, sil = 0, 0
            
            print(f"  eps={eps} min={min_samples}: {n_clusters} clusters, {n_noise} noise | ARI={ari:.4f} Sil={sil:.4f}")
    
    # 5. PCA + t-SNE for visualization
    print(f"\n{'='*70}")
    print("  5. DIMENSIONALITY REDUCTION")
    print(f"{'='*70}")
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.astype(float))
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
    
    results["pca"] = {
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "total_variance": round(float(sum(pca.explained_variance_ratio_)), 4),
    }
    
    # t-SNE (subsample if large)
    n_tsne = min(5000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_tsne, replace=False)
    X_sub = X[idx].astype(float)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_sub)
    print(f"  t-SNE computed on {n_tsne} samples")
    
    # 6. Feature importance via mutual information
    print(f"\n{'='*70}")
    print("  6. FEATURE IMPORTANCE (Mutual Information)")
    print(f"{'='*70}")
    
    from sklearn.feature_selection import mutual_info_classif
    
    mi_atk = mutual_info_classif(X, y_atk, discrete_features=True, random_state=42)
    mi_cls = mutual_info_classif(X, y_cls, discrete_features=True, random_state=42)
    mi_tri = mutual_info_classif(X, y_tri, discrete_features=True, random_state=42)
    
    feat_importance = {}
    print(f"\n  {'Feature':<25} {'MI(Cls)':>8} {'MI(Tri)':>8} {'MI(Atk)':>8}")
    print(f"  {'-'*55}")
    for i, fname in enumerate(feat_keys):
        print(f"  {fname:<25} {mi_cls[i]:>8.4f} {mi_tri[i]:>8.4f} {mi_atk[i]:>8.4f}")
        feat_importance[fname] = {
            "mi_classification": round(float(mi_cls[i]), 4),
            "mi_triage": round(float(mi_tri[i]), 4),
            "mi_attack_category": round(float(mi_atk[i]), 4),
        }
    
    results["feature_importance"] = feat_importance
    
    # 7. Unique patterns analysis
    print(f"\n{'='*70}")
    print("  7. UNIQUE PATTERN ANALYSIS")
    print(f"{'='*70}")
    
    patterns = {}
    for i, (feats, labels) in enumerate(zip(all_feats, all_labels)):
        key = tuple(feats.get(k, "?") for k in feat_keys)
        atk = labels.get("attack_category", "?")
        if key not in patterns:
            patterns[key] = {"count": 0, "attack_categories": Counter()}
        patterns[key]["count"] += 1
        patterns[key]["attack_categories"][atk] += 1
    
    # Check ambiguity
    ambiguous = sum(1 for p in patterns.values() if len(p["attack_categories"]) > 1)
    total_patterns = len(patterns)
    
    print(f"  Total unique patterns: {total_patterns}")
    print(f"  Ambiguous patterns (multiple attack cats): {ambiguous}")
    print(f"  Unambiguous patterns: {total_patterns - ambiguous}")
    
    if ambiguous > 0:
        print(f"\n  Ambiguous patterns:")
        for key, info in sorted(patterns.items(), key=lambda x: -len(x[1]["attack_categories"])):
            if len(info["attack_categories"]) > 1:
                print(f"    {key}: {dict(info['attack_categories'])}")
    
    results["pattern_analysis"] = {
        "total_unique_patterns": total_patterns,
        "ambiguous_patterns": ambiguous,
        "unambiguous_patterns": total_patterns - ambiguous,
        "total_samples": len(all_feats),
    }
    
    # Save results
    out_file = os.path.join(OUT_DIR, "clustering_analysis.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅ Results saved to {out_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("  📊 SUMMARY")
    print(f"{'='*70}")
    print(f"  Best K-Means ARI (vs Attack Category): {results['kmeans_vs_attack_category']['ari']:.4f}")
    print(f"  Best K-Means NMI (vs Attack Category): {results['kmeans_vs_attack_category']['nmi']:.4f}")
    best_k = max(elbow_data, key=lambda x: x["silhouette"])
    print(f"  Optimal K (by Silhouette): {best_k['k']} (sil={best_k['silhouette']:.4f})")
    print(f"  Unique patterns: {total_patterns} | Ambiguous: {ambiguous}")
    print(f"  Most important feature for Attack Category: {max(feat_importance, key=lambda k: feat_importance[k]['mi_attack_category'])}")

if __name__ == "__main__":
    main()

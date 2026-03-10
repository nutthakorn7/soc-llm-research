#!/usr/bin/env python3
"""
General AI Paper: Cross-Domain Entropy Analysis
Downloads datasets, computes entropy, trains DT baseline, prepares LLM data.
"""
import json, os, sys
import numpy as np
from collections import Counter
from math import log2
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

OUT_DIR = "/Users/pop7/Code/Lanta/results/general_ai"
os.makedirs(OUT_DIR, exist_ok=True)

def entropy(labels):
    """Compute label entropy in bits."""
    counts = Counter(labels)
    total = sum(counts.values())
    h = 0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * log2(p)
    return round(h, 4)

def run_dt_baseline(X_train, y_train, X_test, y_test):
    """Train DT and return macro F1."""
    dt = DecisionTreeClassifier(random_state=42, max_depth=20)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return round(f1_score(y_test, y_pred, average='macro', zero_division=0) * 100, 2)

def analyze_dataset(name, texts_train, labels_train, texts_test, labels_test, n_classes):
    """Full analysis: entropy + DT baseline."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    # Entropy
    h = entropy(labels_train)
    print(f"  Classes: {n_classes}")
    print(f"  Train: {len(texts_train)} | Test: {len(texts_test)}")
    print(f"  Entropy: {h} bits")
    
    # Class distribution
    counts = Counter(labels_train)
    for label, count in counts.most_common(5):
        print(f"    {label}: {count} ({count/len(labels_train)*100:.1f}%)")
    if len(counts) > 5:
        print(f"    ... and {len(counts)-5} more classes")
    
    # DT baseline with TF-IDF
    print(f"\n  Training DT (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = tfidf.fit_transform(texts_train)
    X_test = tfidf.transform(texts_test)
    
    dt_f1 = run_dt_baseline(X_train, labels_train, X_test, labels_test)
    print(f"  DT F1 (macro): {dt_f1}%")
    
    return {
        "dataset": name,
        "n_classes": n_classes,
        "n_train": len(texts_train),
        "n_test": len(texts_test),
        "entropy_bits": h,
        "dt_f1": dt_f1,
        "llm_f1": None,  # Fill after LLM training
    }

def prepare_llm_data(name, texts, labels, output_dir, max_samples=5000):
    """Convert to ShareGPT format for LlamaFactory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Subsample if needed
    if len(texts) > max_samples:
        idx = np.random.RandomState(42).choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    
    data = []
    for text, label in zip(texts, labels):
        data.append({
            "conversations": [
                {"from": "system", "value": f"Classify the following text into one of the categories."},
                {"from": "human", "value": f"Text: {text[:500]}"},
                {"from": "gpt", "value": f"Category: {label}"}
            ]
        })
    
    with open(os.path.join(output_dir, f"{name}_train.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} samples to {name}_train.json")
    return len(data)

def main():
    print("=" * 60)
    print("  General AI Paper: Cross-Domain Entropy Analysis")
    print("  'When Do You Need an LLM?'")
    print("=" * 60)
    
    results = []
    
    # --- Dataset 0: SALAD (already have results) ---
    salad_result = {
        "dataset": "SALAD (Cybersecurity)",
        "n_classes": 8,
        "n_train": 5000,
        "n_test": 9851,
        "entropy_bits": 2.417,
        "dt_f1": 87.4,
        "llm_f1": 100.0,
    }
    results.append(salad_result)
    print(f"\n  SALAD: Using existing results (DT=87.4%, LLM=100%)")
    
    # --- Dataset 1: AG News ---
    try:
        from datasets import load_dataset
        print("\n  Loading AG News...")
        ag = load_dataset("ag_news", trust_remote_code=True)
        ag_labels_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        
        texts_train = [x["text"] for x in ag["train"]]
        labels_train = [ag_labels_map[x["label"]] for x in ag["train"]]
        texts_test = [x["text"] for x in ag["test"]]
        labels_test = [ag_labels_map[x["label"]] for x in ag["test"]]
        
        r = analyze_dataset("AG News (4-class)", texts_train, labels_train, texts_test, labels_test, 4)
        results.append(r)
        prepare_llm_data("ag_news", texts_train, labels_train, os.path.join(OUT_DIR, "data"))
    except Exception as e:
        print(f"  ❌ AG News failed: {e}")
    
    # --- Dataset 2: GoEmotions (instead of MIMIC-III) ---
    try:
        print("\n  Loading GoEmotions...")
        ge = load_dataset("google-research-datasets/go_emotions", "simplified", trust_remote_code=True)
        emotion_map = {0:'admiration',1:'amusement',2:'anger',3:'annoyance',4:'approval',
                      5:'caring',6:'confusion',7:'curiosity',8:'desire',9:'disappointment',
                      10:'disapproval',11:'disgust',12:'embarrassment',13:'excitement',
                      14:'fear',15:'gratitude',16:'grief',17:'joy',18:'love',19:'nervousness',
                      20:'optimism',21:'pride',22:'realization',23:'relief',24:'remorse',
                      25:'sadness',26:'surprise',27:'neutral'}
        
        # Filter to single-label only
        train_texts, train_labels = [], []
        for x in ge["train"]:
            if len(x["labels"]) == 1:
                train_texts.append(x["text"])
                train_labels.append(emotion_map[x["labels"][0]])
        
        test_texts, test_labels = [], []
        for x in ge["test"]:
            if len(x["labels"]) == 1:
                test_texts.append(x["text"])
                test_labels.append(emotion_map[x["labels"][0]])
        
        n_classes = len(set(train_labels))
        r = analyze_dataset(f"GoEmotions ({n_classes}-class)", train_texts, train_labels, test_texts, test_labels, n_classes)
        results.append(r)
        prepare_llm_data("go_emotions", train_texts, train_labels, os.path.join(OUT_DIR, "data"))
    except Exception as e:
        print(f"  ❌ GoEmotions failed: {e}")
    
    # --- Dataset 3: LEDGAR (Legal) ---
    try:
        print("\n  Loading LEDGAR...")
        led = load_dataset("lex_glue", "ledgar", trust_remote_code=True)
        
        # Get label names
        label_names = led["train"].features["label"].names
        
        texts_train = [x["text"][:500] for x in led["train"]]
        labels_train = [label_names[x["label"]] for x in led["train"]]
        texts_test = [x["text"][:500] for x in led["test"]]
        labels_test = [label_names[x["label"]] for x in led["test"]]
        
        n_classes = len(set(labels_train))
        r = analyze_dataset(f"LEDGAR Legal ({n_classes}-class)", texts_train, labels_train, texts_test, labels_test, n_classes)
        results.append(r)
        prepare_llm_data("ledgar", texts_train, labels_train, os.path.join(OUT_DIR, "data"))
    except Exception as e:
        print(f"  ❌ LEDGAR failed: {e}")
    
    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Entropy vs DT F1")
    print(f"{'='*60}")
    print(f"\n  {'Dataset':<30} {'Classes':>8} {'Entropy':>8} {'DT F1':>8} {'LLM F1':>8}")
    print(f"  {'-'*64}")
    for r in sorted(results, key=lambda x: x['entropy_bits']):
        llm = f"{r['llm_f1']:.1f}%" if r['llm_f1'] else "TBD"
        print(f"  {r['dataset']:<30} {r['n_classes']:>8} {r['entropy_bits']:>7.3f}b {r['dt_f1']:>7.1f}% {llm:>8}")
    
    # Save
    with open(os.path.join(OUT_DIR, "cross_domain_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅ Saved to {OUT_DIR}/cross_domain_results.json")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ICL (In-Context Learning) baselines for Q1 Rule of Law.
Uses Gemini, GPT-4o-mini, and Claude via API.
"""
import json, os, sys, time
from dotenv import load_dotenv
from sklearn.metrics import f1_score

load_dotenv()

def load_test_samples(path, n=500):
    with open(path) as f:
        data = json.load(f)
    samples = data[:n]
    texts, labels = [], []
    for item in samples:
        convs = item.get("conversations", [])
        inp, out = "", ""
        for c in convs:
            if c["from"] == "human": inp = c["value"]
            elif c["from"] == "gpt": out = c["value"]
        
        # Extract Attack Category label
        label = ""
        for line in out.split("\n"):
            if "Attack Category:" in line:
                label = line.split(":", 1)[1].strip()
                break
        if not label:
            label = out.strip()
        
        texts.append(inp[:1500])
        labels.append(label)
    return texts, labels

def build_prompt(text, classes):
    return f"""Classify the following text into one of these categories: {', '.join(classes)}

Text: {text}

Respond with ONLY the category name, nothing else."""

def call_gemini(prompt):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()

def call_openai(prompt):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50, temperature=0
    )
    return resp.choices[0].message.content.strip()

def call_claude(prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text.strip()

APIS = {
    "gemini": call_gemini,
    "gpt4o-mini": call_openai,
    "claude": call_claude,
}

def run_icl(dataset_path, model_name, classes, n=500):
    texts, labels = load_test_samples(dataset_path, n)
    
    call_fn = APIS[model_name]
    preds = []
    errors = 0
    
    for i, text in enumerate(texts):
        prompt = build_prompt(text, classes)
        try:
            pred = call_fn(prompt)
            preds.append(pred)
        except Exception as e:
            preds.append("")
            errors += 1
            if errors > 10:
                print(f"Too many errors ({errors}), stopping")
                break
        
        if (i+1) % 50 == 0:
            print(f"  {model_name}: {i+1}/{len(texts)} done ({errors} errors)")
        time.sleep(0.5)  # rate limit
    
    # Truncate labels to match
    labels = labels[:len(preds)]
    
    # Simple matching
    correct = sum(1 for p, l in zip(preds, labels) if p.lower().strip() in l.lower() or l.lower().strip() in p.lower())
    acc = correct / len(preds)
    
    print(f"\n  {model_name} on {os.path.basename(dataset_path)}:")
    print(f"    Accuracy: {acc*100:.1f}%")
    print(f"    Errors: {errors}/{len(texts)}")
    
    # Save results
    out_dir = f"results/icl-{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{os.path.basename(dataset_path)}_results.json", "w") as f:
        json.dump({
            "model": model_name, "dataset": dataset_path,
            "accuracy": acc, "n": len(preds), "errors": errors,
            "predictions": preds[:20],  # sample
        }, f, indent=2)
    
    return acc

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    datasets = {
        "SALAD": ("data/test_held_out.json", [
            "Reconnaissance", "DoS", "Exploits", "Fuzzers",
            "Analysis", "Backdoor", "Generic", "Benign"
        ]),
    }
    
    print(f"=== ICL Baseline: {model} (n={n}) ===")
    for name, (path, classes) in datasets.items():
        if os.path.exists(path):
            run_icl(path, model, classes, n)
        else:
            print(f"  {name}: file not found ({path})")

#!/usr/bin/env python3
"""
BERT-base classification baseline for Q1 Rule of Law compliance.
Uses HuggingFace Trainer (not LlamaFactory) since BERT is not generative.
"""
import json, re, sys, os
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_sharegpt(path, task="attack_category"):
    """Load ShareGPT format and extract labels."""
    with open(path) as f:
        data = json.load(f)
    
    texts, labels = [], []
    for item in data:
        convs = item.get("conversations", [])
        inp = ""
        out = ""
        for c in convs:
            if c["from"] == "human": inp = c["value"]
            elif c["from"] == "gpt": out = c["value"]
        
        label = out.strip()
        if "Attack Category:" in out:
            for line in out.split("\n"):
                if "Attack Category:" in line:
                    label = line.split(":", 1)[1].strip()
                    break
        elif label.startswith("Category: "):
            label = label[10:]
        
        texts.append(inp[:512])
        labels.append(label[:80])
    
    return texts, labels


def train_bert(texts, labels, output_dir, epochs=5, batch_size=16):
    """Train BERT-base for classification."""
    from transformers import (
        BertTokenizer, BertForSequenceClassification,
        TrainingArguments, Trainer
    )
    from torch.utils.data import Dataset
    import torch
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_labels = len(le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42
    )
    
    BERT_PATH = "/project/lt200473-ttctvs/soc-finetune/models/bert-base-uncased"
    
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    
    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
    
    train_ds = TextDataset(X_train, y_train.tolist())
    test_ds = TextDataset(X_test, y_test.tolist())
    
    model = BertForSequenceClassification.from_pretrained(
        BERT_PATH, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=True,
        logging_steps=50,
        seed=42,
    )
    
    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=-1)
        f1 = f1_score(pred.label_ids, preds, average="macro", zero_division=0)
        acc = (preds == pred.label_ids).mean()
        return {"f1": f1, "accuracy": acc}
    
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    results = trainer.evaluate()
    
    # Per-class report
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"  BERT-base Results: {output_dir}")
    print(f"{'='*60}")
    print(f"  Macro-F1: {results['eval_f1']:.4f}")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")
    print(f"\n{report}")
    
    # Save results
    with open(os.path.join(output_dir, "bert_results.json"), "w") as f:
        json.dump({
            "macro_f1": results["eval_f1"],
            "accuracy": results["eval_accuracy"],
            "num_classes": num_labels,
            "classes": le.classes_.tolist(),
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/bert-baseline"
    
    print(f"Loading {data_path}...")
    texts, labels = load_sharegpt(data_path)
    print(f"  {len(texts)} samples, {len(set(labels))} classes")
    
    train_bert(texts, labels, output_dir)

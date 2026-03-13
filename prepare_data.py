#!/usr/bin/env python3
"""
Prepare SALAD dataset for LlamaFactory fine-tuning.
Converts SOC alerts into instruction-response format.
"""
import json
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "/project/lt200473-ttctvs/soc-finetune/data"

def alert_to_instruction(row):
    """Convert a SALAD alert row to natural language instruction."""
    parts = []
    if pd.notna(row.get('alert_type')):
        parts.append(f"Alert Type: {row['alert_type']}")
    if pd.notna(row.get('severity')):
        parts.append(f"Severity: {row['severity']}")
    if pd.notna(row.get('source_ip')):
        parts.append(f"Source IP: {row['source_ip']}")
    if pd.notna(row.get('destination_ip')):
        parts.append(f"Destination IP: {row['destination_ip']}")
    if pd.notna(row.get('source_port')):
        parts.append(f"Source Port: {int(row['source_port'])}")
    if pd.notna(row.get('destination_port')):
        parts.append(f"Destination Port: {int(row['destination_port'])}")
    if pd.notna(row.get('protocol')):
        parts.append(f"Protocol: {row['protocol']}")
    if pd.notna(row.get('mitre_tactic')):
        parts.append(f"MITRE Tactic: {row['mitre_tactic']}")
    if pd.notna(row.get('mitre_technique')):
        parts.append(f"MITRE Technique: {row['mitre_technique']}")
    if pd.notna(row.get('kill_chain_phase')):
        parts.append(f"Kill Chain Phase: {row['kill_chain_phase']}")
    if pd.notna(row.get('network_segment')):
        parts.append(f"Network Segment: {row['network_segment']}")
    if pd.notna(row.get('description')):
        parts.append(f"Description: {row['description']}")

    alert_text = "\n".join(parts)
    return alert_text


def alert_to_response(row):
    """Convert SALAD labels to structured response."""
    parts = []
    if pd.notna(row.get('is_malicious')):
        label = "Malicious" if row['is_malicious'] == 1 else "Benign"
        parts.append(f"Classification: {label}")
    if pd.notna(row.get('triage_decision')):
        parts.append(f"Triage Decision: {row['triage_decision']}")
    if pd.notna(row.get('attack_category')):
        parts.append(f"Attack Category: {row['attack_category']}")
    if pd.notna(row.get('priority_score')):
        parts.append(f"Priority Score: {row['priority_score']:.2f}")
    return "\n".join(parts)


SYSTEM_PROMPT = """You are an expert SOC (Security Operations Center) analyst. Given a security alert with network and threat intelligence details, provide:
1. Classification: Malicious or Benign
2. Triage Decision: Escalate, Investigate, or Suppress
3. Attack Category: The specific type of attack (e.g., DoS, Reconnaissance, Backdoor, etc.)
4. Priority Score: A score from 0.00 (lowest) to 1.00 (highest)

Respond concisely with only the four fields above."""


def convert_to_llamafactory(df, output_file, max_samples=None):
    """Convert DataFrame to LlamaFactory ShareGPT format."""
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    records = []
    for _, row in df.iterrows():
        instruction = alert_to_instruction(row)
        response = alert_to_response(row)

        record = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": f"Analyze this SOC alert:\n\n{instruction}"},
                {"from": "gpt", "value": response}
            ]
        }
        records.append(record)

    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"  Saved {len(records)} samples to {output_file}")
    return len(records)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Downloading SALAD dataset from HuggingFace ===")
    try:
        ds = load_dataset("nutthakorn7/SALAD-SOC")
        print(f"Dataset loaded: {ds}")

        # Use existing splits
        train_df = ds['train'].to_pandas()
        val_df = ds['validation'].to_pandas() if 'validation' in ds else None
        test_df = ds['test'].to_pandas() if 'test' in ds else None
    except Exception as e:
        print(f"HuggingFace load failed: {e}")
        print("Trying CSV fallback...")
        # Fallback: download CSVs directly
        base_url = "https://huggingface.co/datasets/nutthakorn7/SALAD-SOC/resolve/main"
        train_df = pd.read_csv(f"{base_url}/train.csv")
        val_df = pd.read_csv(f"{base_url}/validation.csv")
        test_df = pd.read_csv(f"{base_url}/test.csv")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df):,}")
    if val_df is not None:
        print(f"  Validation: {len(val_df):,}")
    if test_df is not None:
        print(f"  Test: {len(test_df):,}")

    # Print columns for debugging
    print(f"\nColumns: {list(train_df.columns)}")

    # Convert to LlamaFactory format
    print("\n=== Converting to LlamaFactory format ===")

    # Training: start with 50K samples for efficiency (can scale up later)
    print("Converting training set (50K sample)...")
    n_train = convert_to_llamafactory(
        train_df,
        os.path.join(OUTPUT_DIR, "train_50k.json"),
        max_samples=50000
    )

    # Full training set
    print("Converting full training set...")
    n_train_full = convert_to_llamafactory(
        train_df,
        os.path.join(OUTPUT_DIR, "train_full.json"),
        max_samples=None
    )

    # Validation: 5K sample
    if val_df is not None:
        print("Converting validation set (5K sample)...")
        convert_to_llamafactory(
            val_df,
            os.path.join(OUTPUT_DIR, "val_5k.json"),
            max_samples=5000
        )

    # Test: full test set
    if test_df is not None:
        print("Converting test set (full)...")
        convert_to_llamafactory(
            test_df,
            os.path.join(OUTPUT_DIR, "test_full.json"),
            max_samples=None
        )

    # Create dataset_info.json for LlamaFactory
    dataset_info = {
        "salad_50k": {
            "file_name": os.path.join(OUTPUT_DIR, "train_50k.json"),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        },
        "salad_full": {
            "file_name": os.path.join(OUTPUT_DIR, "train_full.json"),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        },
        "salad_val": {
            "file_name": os.path.join(OUTPUT_DIR, "val_5k.json"),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        },
        "salad_test": {
            "file_name": os.path.join(OUTPUT_DIR, "test_full.json"),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        }
    }

    info_path = os.path.join(OUTPUT_DIR, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\nSaved dataset_info.json to {info_path}")

    # Summary
    print("\n=== Data Preparation Complete ===")
    print(f"Training (50K):  {os.path.join(OUTPUT_DIR, 'train_50k.json')}")
    print(f"Training (Full): {os.path.join(OUTPUT_DIR, 'train_full.json')}")
    print(f"Validation (5K): {os.path.join(OUTPUT_DIR, 'val_5k.json')}")
    print(f"Test (Full):     {os.path.join(OUTPUT_DIR, 'test_full.json')}")


if __name__ == "__main__":
    main()

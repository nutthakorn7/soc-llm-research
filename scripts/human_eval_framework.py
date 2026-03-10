#!/usr/bin/env python3
"""
Human Evaluation Framework for SOC-FT Paper
Generates survey materials for SOC analyst evaluation.

PURPOSE:
- Ask real SOC analysts to rate LLM vs DT explanations
- Proves LLM adds value beyond accuracy (interpretability + actionability)
- Required for Tier 1 IEEE venue
"""
import json, os, random

BASE = "/project/lt200473-ttctvs/soc-finetune"
OUT = os.path.join(BASE, "outputs/paper_results/human_eval")
os.makedirs(OUT, exist_ok=True)

# Survey design
SURVEY = {
    "title": "SOC Alert Triage Evaluation: LLM vs Traditional ML",
    "description": """
    You will be shown 30 SOC alerts with 2 different triage outputs:
    - System A: Label only (e.g., "Attack Category: reconnaissance_scan")
    - System B: Label + natural language explanation
    
    For each alert, please rate both systems on a 1-5 scale for:
    1. Correctness: Is the triage decision correct?
    2. Actionability: Can you act on this information immediately?
    3. Trust: Would you trust this in production?
    4. Speed: How quickly could you make a decision with this output?
    """,
    "demographics": [
        "Years of SOC experience",
        "Current role (Tier 1/2/3/Manager)",
        "Certifications (CISSP, CEH, GCIH, etc.)",
        "Average daily alert volume",
    ],
    "rating_scale": {
        1: "Very Poor",
        2: "Poor", 
        3: "Acceptable",
        4: "Good",
        5: "Excellent"
    },
    "metrics": [
        {"name": "Correctness", "question": "Is the classification/triage decision accurate?"},
        {"name": "Actionability", "question": "Can you immediately determine what action to take?"},
        {"name": "Trust", "question": "Would you trust this output in a production SOC?"},
        {"name": "Completeness", "question": "Does the output provide sufficient context?"},
        {"name": "Preference", "question": "Which system do you prefer? (A/B/No preference)"},
    ],
    "sample_size": {
        "minimum": 10,  # analysts
        "recommended": 20,
        "alerts_per_analyst": 30,
        "total_ratings": "20 × 30 × 4 metrics = 2,400 data points"
    },
    "statistical_tests": [
        "Paired t-test (System A vs B per metric)",
        "Wilcoxon signed-rank test (non-parametric)",
        "Cohen's kappa (inter-rater agreement)",
        "Fleiss' kappa (multi-rater agreement)",
    ]
}

def generate_sample_pairs():
    """Generate DT vs LLM output pairs for the survey."""
    examples = [
        {
            "alert": {
                "alert_type": "exploit_attempt",
                "severity": "High",
                "protocol": "TCP",
                "mitre_tactic": "Initial Access",
                "mitre_technique": "T1190",
                "network_segment": "DMZ"
            },
            "system_a": {
                "classification": "Malicious",
                "triage": "Escalate",
                "attack_category": "web_application_attack"
            },
            "system_b": {
                "classification": "Malicious",
                "triage": "Escalate",
                "attack_category": "web_application_attack",
                "priority_score": 8.5,
                "explanation": "High-severity exploit attempt targeting DMZ via TCP. "
                    "MITRE T1190 (Exploit Public-Facing Application) indicates active "
                    "exploitation of web services. Immediate investigation of DMZ web "
                    "servers recommended. Check WAF logs and application access logs "
                    "for payload analysis."
            }
        },
        {
            "alert": {
                "alert_type": "normal_traffic",
                "severity": "Low",
                "protocol": "UDP",
                "mitre_tactic": "N/A",
                "mitre_technique": "N/A",
                "network_segment": "Internal"
            },
            "system_a": {
                "classification": "Benign",
                "triage": "Suppress",
                "attack_category": "none"
            },
            "system_b": {
                "classification": "Benign",
                "triage": "Suppress",
                "attack_category": "none",
                "priority_score": 1.0,
                "explanation": "Routine internal UDP traffic with no MITRE mapping. "
                    "Low severity, consistent with normal network operations. "
                    "No action required. Safe to auto-suppress."
            }
        },
        {
            "alert": {
                "alert_type": "reconnaissance_scan",
                "severity": "Medium",
                "protocol": "TCP",
                "mitre_tactic": "Discovery",
                "mitre_technique": "T1046",
                "network_segment": "External"
            },
            "system_a": {
                "classification": "Malicious",
                "triage": "Investigate",
                "attack_category": "network_scan"
            },
            "system_b": {
                "classification": "Malicious",
                "triage": "Investigate",
                "attack_category": "network_scan",
                "priority_score": 5.5,
                "explanation": "TCP port scan from external source. MITRE T1046 "
                    "(Network Service Discovery) suggests adversary reconnaissance. "
                    "Medium priority — may be automated scanning or targeted recon. "
                    "Check if source IP appears in threat intel feeds. Block if "
                    "targeting critical services."
            }
        }
    ]
    return examples

def main():
    print("=" * 60)
    print("  HUMAN EVALUATION FRAMEWORK")
    print("=" * 60)
    
    # Save survey design
    with open(os.path.join(OUT, "survey_design.json"), "w") as f:
        json.dump(SURVEY, f, indent=2)
    
    # Save sample pairs
    pairs = generate_sample_pairs()
    with open(os.path.join(OUT, "sample_pairs.json"), "w") as f:
        json.dump(pairs, f, indent=2)
    
    print(f"\n  Survey Design:")
    print(f"    Analysts needed: {SURVEY['sample_size']['recommended']}")
    print(f"    Alerts per analyst: {SURVEY['sample_size']['alerts_per_analyst']}")
    print(f"    Total data points: {SURVEY['sample_size']['total_ratings']}")
    print(f"    Metrics: {len(SURVEY['metrics'])}")
    print(f"    Statistical tests: {len(SURVEY['statistical_tests'])}")
    
    print(f"\n  Output files:")
    print(f"    {OUT}/survey_design.json")
    print(f"    {OUT}/sample_pairs.json")
    
    print(f"\n  📋 Next Steps:")
    print(f"    1. Recruit 10-20 SOC analysts (Tier 1-3)")
    print(f"    2. Create Google Form / Qualtrics survey")
    print(f"    3. Randomize System A/B presentation (blind)")
    print(f"    4. Collect ratings for 30 alerts × 4 metrics")
    print(f"    5. Run paired t-test + Cohen's kappa")

if __name__ == "__main__":
    main()

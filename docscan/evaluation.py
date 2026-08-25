"""Evaluate extraction accuracy against ground truth"""
import json
from pathlib import Path
from difflib import SequenceMatcher

def load_ground_truth(json_path: str) -> dict:
    """Load ground truth labels"""
    with open(json_path, "r") as f:
        return json.load(f)

def string_similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings (0.0 to 1.0)"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def evaluate_extraction(pdf_name: str, extracted: dict, ground_truth: dict) -> dict:
    """
    Compare extracted fields vs ground truth.
    Returns accuracy metrics.
    """
    if pdf_name not in ground_truth:
        raise ValueError(f"No ground truth for {pdf_name}")
    
    truth = ground_truth[pdf_name]
    
    results = {}
    
    # Check each field
    for field_name, true_value in truth.items():
        extracted_value = extracted.get(field_name, None)
        
        # For emails, check if both are in list
        if field_name == "emails":
            if isinstance(extracted_value, str):
                extracted_value = [e.strip() for e in extracted_value.split(",")]
            match = len([e for e in (extracted_value or []) if e in true_value])
            accuracy = match / len(true_value) if true_value else 0.0
        else:
            # For text fields, use string similarity
            accuracy = string_similarity(extracted_value, true_value)
        
        results[field_name] = {
            "extracted": extracted_value,
            "ground_truth": true_value,
            "accuracy": round(accuracy, 2)
        }
    
    # Overall accuracy
    overall = sum(r["accuracy"] for r in results.values()) / len(results)
    results["overall_accuracy"] = round(overall, 2)
    
    return results

if __name__ == "__main__":
    # Test it
    truth = load_ground_truth("data/ground_truth.json")
    
    # Simulate extracted fields (the bad ones we got)
    extracted = {
        "invoice_number": "INV-3337",
        "date": "30 days",  # Wrong!
        "amount": "# 4321 432 Payment",  # Wrong!
        "emails": "admin@slicedinvoices.com, test@test.com",  # Correct
        "from": "DEMO - Sliced Invoices",  # Correct
        "to": "Test Business"  # Need to check if it's in ground truth
    }
    
    results = evaluate_extraction("test.pdf", extracted, truth)
    
    print("\nEvaluation Results:\n")
    for field, metrics in results.items():
        if field == "overall_accuracy":
            print(f"Overall Accuracy: {metrics}")
        else:
            print(f"{field}:")
            print(f"  Extracted: {metrics['extracted']}")
            print(f"  Truth:     {metrics['ground_truth']}")
            print(f"  Accuracy:  {metrics['accuracy']}")

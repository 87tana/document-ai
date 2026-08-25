"""Step 5: Run pipeline (classify -> route -> extract) and log to MLflow"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlflow
import json
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import re
from docscan.evaluation import load_ground_truth, evaluate_extraction
from docscan.classifier import classify_document
from docscan.extraction import extract_invoice_fields, extract_referral_fields

def extract_text_ocr(pdf_path: str) -> str:
    """Extract text using OCR"""
    ocr = PaddleOCR(use_gpu=False, lang='en')
  
    images = convert_from_path(pdf_path)

    all_text = []
    for page_num, image in enumerate(images):
        result = ocr.ocr(np.array(image))
        page_text = ""
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                page_text += text + " "
        all_text.append(page_text)

    return "\n---PAGE BREAK---\n".join(all_text)


    
def run_pipeline_for_document(pdf_name: str, pdf_path: str, truth: dict):
    """Run OCR -> classify -> route -> extract -> evaluate for one document"""
    with mlflow.start_run(run_name=f"pipeline-{pdf_name}"):
        mlflow.log_param("ocr_lang", "en")
        mlflow.log_param("ocr_gpu", False)
        mlflow.log_param("pdf_name", pdf_name)

        ocr_text = extract_text_ocr(pdf_path)

        # Classify
        doc_type = classify_document(ocr_text)
        mlflow.log_param("classified_as", doc_type)
        print(f"  Classified '{pdf_name}' as: {doc_type}")

        # Route to correct extractor
        if doc_type == "invoice":
            extracted = extract_invoice_fields(ocr_text)
            mlflow.log_param("extraction_method", "invoice-label-regex")
        elif doc_type == "referral":
            extracted = extract_referral_fields(ocr_text)
            mlflow.log_param("extraction_method", "referral-label-regex")
        else:
            print(f"  WARNING: unknown document type for {pdf_name}, skipping extraction")
            mlflow.log_metric("overall_accuracy", 0.0)
            return

        # Evaluate
        results = evaluate_extraction(pdf_name, extracted, truth)

        mlflow.log_metric("overall_accuracy", results["overall_accuracy"])
        for field_name, metrics in results.items():
            if field_name != "overall_accuracy":
                mlflow.log_metric(f"{field_name}_accuracy", metrics["accuracy"])

        artifact_path = f"extracted_{pdf_name}.json"
        with open(artifact_path, "w") as f:
            json.dump(extracted, f, indent=2)
        mlflow.log_artifact(artifact_path)

        print(f"  Overall Accuracy: {results['overall_accuracy']}")


if __name__ == "__main__":
    mlflow.set_experiment("docscan-phase0-multidoc")

    truth = load_ground_truth("data/ground_truth.json")

    documents = [
        ("test.pdf", "data/sample_invoices/test.pdf"),
        ("referral.pdf", "data/sample_invoices/referral.pdf"),
    ]

    for pdf_name, pdf_path in documents:
        print(f"\nProcessing: {pdf_name}")
        run_pipeline_for_document(pdf_name, pdf_path, truth)

    print("\n✓ All documents logged to MLflow!")

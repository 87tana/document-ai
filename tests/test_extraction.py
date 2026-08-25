"""Fast unit tests for extraction logic - no OCR, no MLflow.
These run on every CI push to catch regressions in parsing logic.
"""
from docscan.extraction import extract_invoice_fields, extract_referral_fields

def test_invoice_extraction():
    sample_text = (
        "Invoice Number  INV-3337 Invoice Date  January 25, 2016 "
        "Total Due $93.50 admin@slicedinvoices.com"
    )
    result = extract_invoice_fields(sample_text)

    assert result["invoice_number"] == "INV-3337"
    assert result["date"] == "January 25, 2016"
    assert result["amount"] == "$93.50"
    assert "admin@slicedinvoices.com" in result["emails"]

def test_referral_extraction():
    sample_text = (
        "Referral Date: March 10, 2026 Patient Reference Number: REF-8842 "
        "Diagnosis: Suspected rotator cuff injury dr.schmidt@musterpraxis.de"
    )
    result = extract_referral_fields(sample_text)

    assert result["reference_number"] == "REF-8842"
    assert result["referral_date"] == "March 10, 2026"
    assert "dr.schmidt@musterpraxis.de" in result["emails"]

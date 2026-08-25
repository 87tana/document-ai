"""Field extraction logic per document type — label-based regex.
Kept separate from the MLflow experiment script so it's independently testable
(no OCR or MLflow needed to test the parsing logic itself).
"""
import re

def extract_invoice_fields(text: str) -> dict:
    """Extract fields for invoice documents using label-based regex"""
    fields = {
        "invoice_number": None,
        "date": None,
        "amount": None,
        "emails": [],
    }

    inv_match = re.search(r'INV-\d+', text)
    if inv_match:
        fields["invoice_number"] = inv_match.group()

    date_pattern = r'Invoice Date\s+([A-Za-z]+ \d{1,2},\s*\d{4})'
    date_match = re.search(date_pattern, text)
    if date_match:
        fields["date"] = date_match.group(1)

    amount_pattern = r'Total Due\s+(\$[\d,]+\.\d{2})'
    amount_match = re.search(amount_pattern, text)
    if amount_match:
        fields["amount"] = amount_match.group(1)

    fields["emails"] = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return fields

def extract_referral_fields(text: str) -> dict:
    """Extract fields for referral documents using label-based regex"""
    fields = {
        "document_type": "referral",
        "reference_number": None,
        "referral_date": None,
        "diagnosis": None,
        "from_doctor": None,
        "to_doctor": None,
        "emails": [],
    }

    ref_match = re.search(r'REF-\d+', text)
    if ref_match:
        fields["reference_number"] = ref_match.group()

    date_pattern = r'Referral Date\s*:?\s*([A-Za-z]+ \d{1,2},\s*\d{4})'
    date_match = re.search(date_pattern, text)
    if date_match:
        fields["referral_date"] = date_match.group(1)

    diagnosis_pattern = r'Diagnosis\s*:?\s*(.+?)(?:Reason for Referral|$)'
    diagnosis_match = re.search(diagnosis_pattern, text)
    if diagnosis_match:
        fields["diagnosis"] = diagnosis_match.group(1).strip()

    from_pattern = r'From\s*:?\s*(Dr\.\s*[A-Za-z]+\s+[A-Za-z]+)'
    from_match = re.search(from_pattern, text)
    if from_match:
        fields["from_doctor"] = from_match.group(1)

    to_pattern = r'To\s*:?\s*(Dr\.\s*[A-Za-z]+\s+[A-Za-z]+)'
    to_match = re.search(to_pattern, text)
    if to_match:
        fields["to_doctor"] = to_match.group(1)

    fields["emails"] = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return fields

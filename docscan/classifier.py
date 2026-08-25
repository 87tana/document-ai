"""Rule-based document type classifier (Phase 0.2 baseline).
Later this will be replaced by a trained model - same function signature,
so the pipeline around it doesn't need to change.
"""

def classify_document(text: str) -> str:
    """Classify document type based on keyword presence.
    Returns: 'invoice', 'referral', or 'unknown'
    """
    text_lower = text.lower()

    invoice_signals = ["invoice", "total due", "invoice number"]
    referral_signals = ["referral", "diagnosis", "patient reference"]

    invoice_score = sum(1 for kw in invoice_signals if kw in text_lower)
    referral_score = sum(1 for kw in referral_signals if kw in text_lower)

    if invoice_score > referral_score:
        return "invoice"
    elif referral_score > invoice_score:
        return "referral"
    else:
        return "unknown"

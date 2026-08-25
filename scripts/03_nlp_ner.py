"""Step 3: Extract structured fields using NLP/NER"""
import spacy
import re
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

def extract_text_ocr(pdf_path: str) -> str:
    """Convert PDF to images and run OCR"""
    ocr = PaddleOCR(use_gpu=False, lang='en')
    images = convert_from_path(pdf_path)
    print(f"Converted {len(images)} page(s)")
    
    all_text = []
    for page_num, image in enumerate(images):
        print(f"  Processing page {page_num + 1}...")
        result = ocr.ocr(np.array(image))
        
        page_text = ""
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                page_text += text + " "
        
        all_text.append(page_text)
    
    return "\n---PAGE BREAK---\n".join(all_text)

def extract_fields(text: str) -> dict:
    """Extract invoice fields using spaCy NER"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    fields = {
        "invoice_number": None,
        "date": None,
        "amount": None,
        "organizations": [],
        "emails": [],
    }
    
    for ent in doc.ents:
        if ent.label_ == "DATE":
            fields["date"] = ent.text
        elif ent.label_ == "MONEY":
            fields["amount"] = ent.text
        elif ent.label_ == "ORG":
            fields["organizations"].append(ent.text)
    
    inv_match = re.search(r'INV-\d+', text)
    if inv_match:
        fields["invoice_number"] = inv_match.group()
    
    email_match = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    fields["emails"] = email_match
    
    return fields

if __name__ == "__main__":
    pdf_path = "data/sample_invoices/test.pdf"
    text = extract_text_ocr(pdf_path)
    
    fields = extract_fields(text)
    print("\nExtracted Fields:\n")
    for key, value in fields.items():
        print(f"  {key}: {value}")

"""Step 4: QC UI - Show extracted fields and let user edit/approve"""
import streamlit as st
import json
from pathlib import Path
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import spacy
import re

def extract_text_ocr(pdf_path: str) -> str:
    """Convert PDF to images and run OCR"""
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

def extract_fields(text: str) -> dict:
    """Extract invoice fields"""
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

# Streamlit UI
st.title("📄 DocScan QC - Invoice Extraction")

# Upload PDF
uploaded_file = st.file_uploader("Upload invoice PDF", type="pdf")

if uploaded_file:
    # Save temp file
    temp_pdf = "temp_upload.pdf"
    with open(temp_pdf, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✓ Uploaded: {uploaded_file.name}")
    
    # Extract
    with st.spinner("Running OCR..."):
        ocr_text = extract_text_ocr(temp_pdf)
    
    with st.spinner("Extracting fields..."):
        fields = extract_fields(ocr_text)
    
    # Show extracted fields for editing
    st.subheader("Extracted Fields")
    
    edited_fields = {}
    col1, col2 = st.columns(2)
    
    with col1:
        edited_fields["invoice_number"] = st.text_input(
            "Invoice Number", 
            value=fields["invoice_number"] or ""
        )
        edited_fields["date"] = st.text_input(
            "Date", 
            value=fields["date"] or ""
        )
    
    with col2:
        edited_fields["amount"] = st.text_input(
            "Amount", 
            value=fields["amount"] or ""
        )
        edited_fields["emails"] = st.text_input(
            "Emails (comma-separated)", 
            value=", ".join(fields["emails"]) or ""
        )
    
    # Save button
    if st.button("✓ Save & Approve"):
        output_path = Path("data/outputs") / f"{edited_fields['invoice_number']}.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(edited_fields, f, indent=2)
        
        st.success(f"✓ Saved to {output_path}")

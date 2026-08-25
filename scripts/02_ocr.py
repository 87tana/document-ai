"""Step 2: Extract text from PDF using OCR"""
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import numpy as np

def extract_text_ocr(pdf_path: str) -> str:
    """Convert PDF to images and run OCR"""
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_gpu=False, lang='en')
    
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)
    print(f"Converted {len(images)} page(s)")
    
    # Extract text from each page
    all_text = []
    for page_num, image in enumerate(images):
        print(f"  Processing page {page_num + 1}...")
        result = ocr.ocr(np.array(image))
        
        # Extract text from OCR result
        page_text = ""
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                page_text += text + " "
        
        all_text.append(page_text)
    
    return "\n---PAGE BREAK---\n".join(all_text)

if __name__ == "__main__":
    pdf_path = "data/sample_invoices/test.pdf"
    text = extract_text_ocr(pdf_path)
    print("\nExtracted Text:\n")
    print(text[:500])  # Print first 500 chars

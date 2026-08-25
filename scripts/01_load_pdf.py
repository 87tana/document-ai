"""Step 1: Load and validate a PDF file"""
from pathlib import Path

def load_pdf(pdf_path: str) -> Path:
    """Load a PDF and check it exists"""
    path = Path(pdf_path)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"Not a PDF: {pdf_path}")
    
    size_kb = path.stat().st_size / 1024
    print(f"✓ Loaded: {path.name} ({size_kb:.1f} KB)")
    return path

if __name__ == "__main__":
    test_pdf = "data/sample_invoices/test.pdf"
    load_pdf(test_pdf)

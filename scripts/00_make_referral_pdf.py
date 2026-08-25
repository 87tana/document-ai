"""One-off helper: convert referral text to PDF for testing"""
from PIL import Image, ImageDraw, ImageFont

with open("data/sample_invoices/referral_source.txt", "r") as f:
    lines = [line.rstrip() for line in f]

img = Image.new("RGB", (800, 1000), color="white")
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()

y = 20
for line in lines:
    draw.text((20, y), line, fill="black", font=font)
    y += 20

img.save("data/sample_invoices/referral.pdf", "PDF")
print("✓ Created data/sample_invoices/referral.pdf")

FROM python:3.11-slim-bookworm
# System dependencies: poppler-utils for pdf2image
RUN apt-get update && apt-get install -y \
    poppler-utils \
    build-essential \
    swig \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "scripts/04_qc_streamlit.py", "--server.address=0.0.0.0"]

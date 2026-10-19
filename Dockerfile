FROM python:3.12-slim

# Tesseract OCR engine (English language data included)
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Embedding model and tokenizer downloads are cached in a volume
ENV FASTEMBED_CACHE_PATH=/cache/fastembed \
    TIKTOKEN_CACHE_DIR=/cache/tiktoken \
    PYTHONUNBUFFERED=1

CMD ["python", "cli.py", "--help"]

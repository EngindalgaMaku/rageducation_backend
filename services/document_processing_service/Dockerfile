FROM python:3.11-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem paketlerini güncelle ve gerekli paketleri yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY main.py .

# Port'u ortam değişkeninden al (varsayılan: 8003)
ENV PORT=8003

# Port'u expose et
EXPOSE $PORT

# Sağlık kontrolü ekle
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/health')" || exit 1

# Uygulamayı çalıştır
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port $PORT"]
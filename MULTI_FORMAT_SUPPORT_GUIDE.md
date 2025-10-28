# Multi-Format Document Support Guide

Bu guide, Marker-PDF kütüphanesi ile çoklu belge formatı desteğini etkinleştirmek için gerekli adımları açıklar.

## 📋 Desteklenen Formatlar

**Marker-PDF[full] ile:**

- **PDF** ✅ (Mevcut)
- **DOCX** ✅ (Yeni) - Microsoft Word
- **PPTX** ✅ (Yeni) - Microsoft PowerPoint
- **XLSX** ✅ (Yeni) - Microsoft Excel
- **HTML** ✅ (Teknik destek)
- **EPUB** ✅ (Teknik destek)
- **Images** ✅ (PNG, JPG, etc.)

## 🚀 Deployment Adımları

### 1. Local Development Test

```bash
# Marker format desteğini test et
python -c "
from marker.converters.pdf import PdfConverter
from marker.converters.docx import DocxConverter
from marker.converters.pptx import PptxConverter
print('✅ All converters available!')
"
```

### 2. Dependencies Upgrade

Eğer format desteği eksikse:

```bash
# Upgrade to full version
pip uninstall marker-pdf
pip install marker-pdf[full]>=0.2.15
```

### 3. Docker Build

```bash
# Docker image'ı yeniden build et
docker build -f Dockerfile.api -t rageducation-backend:latest .

# Test build
docker run --rm -p 8080:8080 rageducation-backend:latest
```

### 4. Cloud Run Deployment

```bash
# Build ve deploy
gcloud builds submit --config cloudbuild.yaml

# Veya deploy script ile
chmod +x deploy_to_cloud_run.sh
./deploy_to_cloud_run.sh
```

## 🧪 Test Scenarios

### Format Test Files

1. **PDF Test**: Akademik makale, çoklu sayfa
2. **DOCX Test**: Word belgesi, formatlı metin
3. **PPTX Test**: Sunum, çoklu slide
4. **XLSX Test**: Excel tablosu, veri analizi

### Test Commands

```bash
# API endpoint test
curl -X POST -F "file=@test.docx" \
  https://your-api-url/documents/convert-document-to-markdown

# Frontend test
# Upload different file formats through UI
```

## 📊 Performance Considerations

### Memory Requirements

- **PDF**: ~2GB RAM (büyük dosyalar için)
- **DOCX**: ~1GB RAM
- **PPTX**: ~1GB RAM
- **XLSX**: ~0.5GB RAM

### Processing Times

- **PDF**: 5-15 dakika (sayfa sayısına göre)
- **DOCX**: 1-3 dakika
- **PPTX**: 1-5 dakika
- **XLSX**: 30 saniye - 2 dakika

## 🔧 Troubleshooting

### Common Issues

1. **Missing Converters Error**

   ```
   ImportError: No module named 'marker.converters.docx'
   ```

   **Çözüm**: `pip install marker-pdf[full]`

2. **Memory Issues**

   ```
   OutOfMemoryError during processing
   ```

   **Çözüm**: Cloud Run memory limit artır (16GB+)

3. **Timeout Errors**
   ```
   Processing timeout after 900s
   ```
   **Çözüm**: `MARKER_TIMEOUT_SECONDS=1800` environment variable

### Debug Commands

```bash
# Check installed packages
pip show marker-pdf

# Test specific converter
python -c "from marker.converters.docx import DocxConverter; print('DOCX OK')"

# Memory check during processing
docker stats container_name
```

## 🎯 Success Criteria

### Functionality Tests

- [ ] PDF conversion çalışıyor
- [ ] DOCX conversion çalışıyor
- [ ] PPTX conversion çalışıyor
- [ ] XLSX conversion çalışıyor
- [ ] Frontend tüm formatları kabul ediyor
- [ ] Hata mesajları anlamlı

### Performance Tests

- [ ] Büyük PDF (>50MB) işleniyor
- [ ] Çoklu format aynı anda çalışıyor
- [ ] Memory leakage yok
- [ ] Response time makul (<5 min küçük dosyalar)

### Production Tests

- [ ] Cloud Run'da çalışıyor
- [ ] HTTPS endpoint erişilebilir
- [ ] Error handling doğru
- [ ] Logging düzgün çalışıyor

## 📝 Changelog

### v1.0 - Multi-Format Support

- ✅ Frontend: PDF, DOCX, PPTX, XLSX accept
- ✅ Backend: API endpoint güncellemesi
- ✅ Dependencies: marker-pdf[full] upgrade
- ✅ Docker: Build configuration update
- 🔄 Testing: Format compatibility tests
- 🔄 Deployment: Cloud Run optimization

## 🚨 Critical Notes

1. **Docker Build**: Yeni dependencies cache'lemek için rebuild gerekli
2. **Memory Limits**: Cloud Run için minimum 8GB öneriliyor
3. **Timeout**: Büyük dosyalar için timeout artırılmalı
4. **Cost**: marker-pdf[full] daha fazla dependency, build süresi artabilir

## 📞 Support

Sorun yaşarsanız:

1. Logs kontrolü: `docker logs container_name`
2. Memory monitoring: `docker stats`
3. Test isolated: Tek format ile test
4. Fallback: PDF-only mode'a geçiş mümkün

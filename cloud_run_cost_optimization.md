# Cloud Run Maliyet Optimizasyonu - $75 → $5-15/ay

## ⚠️ Yanlış Anlaşılma

$75.35/ay fiyatı **sürekli çalışan** instance içindir. Cloud Run **serverless** - sadece request geldiğinde çalışır!

## ✅ Optimize Edilmiş Ayarlar

### 1. Resource Azaltması:

```
CPU: 4 vCPU → 2 vCPU
Memory: 8 GiB → 4 GiB
Scale to Zero: Aktif (önemli!)
Min Instances: 0 (önemli!)
Max Instances: 10
```

### 2. Gerçek Kullanım Maliyeti:

#### Hafif Kullanım (günde 20 PDF):

- Request sayısı: ~600/ay
- İşlem süresi: PDF başına 30 saniye
- **Maliyet: $5-8/ay** ✅

#### Orta Kullanım (günde 100 PDF):

- Request sayısı: ~3000/ay
- İşlem süresi: PDF başına 30 saniye
- **Maliyet: $12-18/ay** ✅

#### Yoğun Kullanım (günde 300 PDF):

- Request sayısı: ~9000/ay
- İşlem süresi: PDF başına 30 saniye
- **Maliyet: $25-35/ay** ✅

## 🔧 Maliyet Düşürme Stratejileri

### A) Minimum Configurasyon:

```bash
gcloud run deploy rag3-api \
  --memory 2Gi \        # 4GB yerine 2GB
  --cpu 1 \             # 2 vCPU yerine 1 vCPU
  --min-instances 0 \   # Scale to zero
  --max-instances 3 \   # Daha az max instance
  --timeout 900         # 15 dakika timeout
```

**Tahmini Maliyet: $3-10/ay**

### B) Balanced Configurasyon:

```bash
gcloud run deploy rag3-api \
  --memory 4Gi \        # PDF işleme için yeterli
  --cpu 2 \             # Dengeli performans
  --min-instances 0 \   # Scale to zero
  --max-instances 5 \
  --timeout 1800        # 30 dakika timeout
```

**Tahmini Maliyet: $8-15/ay**

## 💡 Ekstra Tasarruf İpuçları

### 1. Request Timeout Optimization:

- Büyük PDF'ler: 30 dakika
- Küçük PDF'ler: 5 dakika
- Soru-cevap: 30 saniye

### 2. Memory Management:

- PDF Processing: 2-4GB yeterli
- AI Models: 1-2GB
- Vector Operations: 1GB

### 3. Cold Start Azaltma:

```yaml
# Min instance 0 yerine 0.1 CPU always-on
--cpu-throttling=false
--min-instances=0
```

## 📊 Gerçekçi Maliyet Tablosu

| Kullanım Seviyesi | Request/Gün | Aylık Maliyet | Yıllık Maliyet |
| ----------------- | ----------- | ------------- | -------------- |
| **Test/Demo**     | 5-10        | $2-5          | $24-60         |
| **Hafif**         | 20-50       | $8-15         | $96-180        |
| **Orta**          | 100-200     | $20-35        | $240-420       |
| **Yoğun**         | 300-500     | $50-75        | $600-900       |

## 🚀 Önerilen Başlangıç Ayarları

Google Cloud Console'da şu ayarları yapın:

1. **CPU:** 2 vCPU (1 vCPU PDF işleme için yavaş olabilir)
2. **Memory:** 4 GiB (PDF + AI modelleri için)
3. **Min instances:** 0 (çok önemli!)
4. **Max instances:** 3-5
5. **Timeout:** 1800 saniye (30 dakika)

Bu ayarlarla **aylık maliyet $10-20 arası** olacaktır.

## 🔄 Sonradan Ayarlama

Kullanımı izleyip optimize edebilirsiniz:

- Az kullanım → Memory/CPU azalt
- Çok kullanım → Performans için artır
- Cold start problem → Min instance 1 yapın

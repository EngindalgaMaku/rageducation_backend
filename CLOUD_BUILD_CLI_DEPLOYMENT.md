# Google Cloud Build CLI Deployment Guide

Bu kılavuz GitHub trigger'ı yerine gcloud CLI ile manuel deployment yapmak için hazırlanmıştır.

## 🛠️ Ön Gereksinimler

1. **Google Cloud SDK kurulumu:**

```bash
# Windows için
# Google Cloud SDK'yı indirin: https://cloud.google.com/sdk/docs/install

# Mac/Linux için
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

2. **Authentication:**

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

3. **Gerekli API'leri aktifleştir:**

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## 🚀 Deployment Komutları

### 1. Temel Deployment

```bash
# Mevcut dizinde cloudbuild.yaml ile build çalıştır
gcloud builds submit --config cloudbuild.yaml .
```

### 2. Özel Parametrelerle Deployment

```bash
# Servis adı ve region belirterek
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_SERVICE=rageducation-backend,_REGION=europe-west1 \
  .
```

### 3. Build Monitoring

```bash
# Son 5 build'i listele
gcloud builds list --limit=5

# Belirli bir build'in detayları
gcloud builds describe BUILD_ID

# Build loglarını canlı izle
gcloud builds log BUILD_ID --stream
```

### 4. Cloud Run Service Kontrolü

```bash
# Deploy edilen servisi kontrol et
gcloud run services describe rageducation-backend --region=europe-west1

# Servis URL'ini al
gcloud run services describe rageducation-backend --region=europe-west1 --format='value(status.url)'

# Servisi test et
curl "$(gcloud run services describe rageducation-backend --region=europe-west1 --format='value(status.url)')/health"
```

## 📋 Hızlı Deployment Script

Kolay kullanım için `deploy.sh` scripti:

```bash
#!/bin/bash
set -e

PROJECT_ID=$(gcloud config get-value project)
echo "🚀 Deploying to project: $PROJECT_ID"

# Build ve deploy
echo "📦 Starting build..."
gcloud builds submit --config cloudbuild.yaml .

# Service URL'ini al ve göster
echo "🌐 Getting service URL..."
SERVICE_URL=$(gcloud run services describe rageducation-backend --region=europe-west1 --format='value(status.url)')
echo "✅ Deployment complete!"
echo "🔗 Service URL: $SERVICE_URL"

# Health check
echo "🧪 Testing health endpoint..."
curl -s "$SERVICE_URL/health" && echo "✅ API is healthy!"
```

## ⚡ Hızlı Komutlar

```bash
# Tek komutla deploy
gcloud builds submit --config cloudbuild.yaml .

# Build durumunu kontrol et
gcloud builds list --limit=1

# Servisi test et
curl "$(gcloud run services describe rageducation-backend --region=europe-west1 --format='value(status.url)')"

# Log'ları izle
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=rageducation-backend' --limit=50
```

## 🔧 Troubleshooting

### Build Hatası

```bash
# Build loglarını detaylı incele
gcloud builds describe BUILD_ID

# Belirli bir step'in logları
gcloud logging read "resource.type=build AND logName=projects/PROJECT_ID/logs/cloudbuild" --limit=100
```

### Service Hatası

```bash
# Cloud Run service logları
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=rageducation-backend' --limit=50

# Service configuration
gcloud run services describe rageducation-backend --region=europe-west1
```

## 📈 Avantajları

✅ **GitHub trigger problemlerinden kurtulma**  
✅ **Build parametrelerini runtime'da kontrol etme**  
✅ **Local test imkanı**  
✅ **Detaylı build monitoring**  
✅ **Manual control over deployments**

## 🎯 Deployment Workflow

1. **Code değişikliği yap**
2. **Terminal'de projeye git**: `cd path/to/rag3_for_colab`
3. **Deploy et**: `gcloud builds submit --config cloudbuild.yaml .`
4. **Test et**: Service URL'e istek at
5. **Monitor et**: Build ve service durumunu kontrol et

## ⚠️ Önemli Notlar

- GitHub trigger'ı manuel olarak disable etmeyi unutmayın
- Her deployment'da BUILD_ID benzersiz olacaktır
- GCS bucket isimleri BUILD_ID ile unique yapılmıştır
- Build süresi yaklaşık 15-20 dakikadır
- Service cold start süresi ~30 saniyedir (lazy imports sayesinde)

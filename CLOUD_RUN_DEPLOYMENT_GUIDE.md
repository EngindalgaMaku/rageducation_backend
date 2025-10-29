# Windows Kullanıcıları İçin Google Cloud Run Dağıtım Rehberi

## Kopyala-Yapıştır Komut Rehberi

## 🙏 Özür ve Yeni Yaklaşım

**Önceki betik tabanlı dağıtım yöntemi için özür dileriz.** Karmaşık betikler şeffaflığı engelliyor ve hata ayıklamayı zorlaştırıyordu.

Artık **%100 şeffaf, kopyala-yapıştır komut rehberi** kullanıyoruz:

✅ **Her komut görünür ve anlaşılır**  
✅ **Adım adım kontrol edilebilir**  
✅ **Hiçbir gizli işlem yok**  
✅ **Hata durumunda tam şeffaflık**

---

## Bölüm 1: Kurulum (Sadece Bir Kez)

Aşağıdaki komutları Windows Komut İstemi'nde (`cmd.exe`) çalıştırın:

### Adım 1.1: Proje ID'nizi ayarlayın

```cmd
set PROJECT_ID=sizin-google-cloud-proje-id-buraya
```

> **Not:** `sizin-google-cloud-proje-id-buraya` kısmını gerçek Google Cloud proje ID'nizle değiştirin.

### Adım 1.2: Bölgeyi ayarlayın

```cmd
set REGION=europe-west1
```

---

## Bölüm 2: Adım Adım Dağıtım Komutları

Her komutu sırayla kopyalayıp yapıştırın. Her adımın başarılı olmasını bekleyin.

### 1. Artifact Repository Oluştur

```cmd
gcloud artifacts repositories create rag-services --repository-format=docker --location=%REGION% --description="RAG Services Docker repository" --project=%PROJECT_ID%
```

**Ne işe yarar:** Docker imajlarınız için depo oluşturur.

**Beklenen çıktı:** `Created repository [rag-services]` veya `Repository [rag-services] already exists` (normal hata - depo zaten varsa).

---

### 2. PDF-Processor İmajını Oluştur

```cmd
gcloud builds submit ./services/pdf_processing_service --tag %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/pdf-processor --project=%PROJECT_ID%
```

**Ne işe yarar:** PDF işleme servisinin Docker imajını oluşturur.

**Beklenen çıktı:** `SUCCESS` mesajı. Bu adım 5-10 dakika sürer.

---

### 3. PDF-Processor Servisini Dağıt

```cmd
gcloud run deploy pdf-processor --image %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/pdf-processor --platform managed --region %REGION% --no-allow-unauthenticated --cpu=4 --memory=16Gi --project=%PROJECT_ID%
```

**Ne işe yarar:** PDF işleme servisini Cloud Run'da çalıştırır (private servis).

**Beklenen çıktı:** `Service [pdf-processor] revision [pdf-processor-xxxxx] has been deployed and is serving 100 percent of traffic.`

---

### 4. Model-Inferencer İmajını Oluştur

```cmd
gcloud builds submit ./services/model_inference_service --tag %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/model-inferencer --project=%PROJECT_ID%
```

**Ne işe yarar:** Model çıkarım ve embedding servisinin Docker imajını oluşturur. Bu servis artık hem LLM çıkarımı hem de embedding üretimi yapabilir. İmaj oluşturma sırasında `llama3:8b`, `mistral:7b` ve `nomic-embed-text` modelleri indirilir.

**Beklenen çıktı:** `SUCCESS` mesajı. Bu adım 8-15 dakika sürer (embedding modeli eklenmesi nedeniyle biraz daha uzun sürebilir).

---

### 5. Model-Inferencer Servisini Dağıt

```cmd
gcloud run deploy model-inferencer --image %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/model-inferencer --platform managed --region %REGION% --no-allow-unauthenticated --cpu=4 --memory=16Gi --project=%PROJECT_ID%
```

**Ne işe yarar:** Model çıkarım ve embedding servisini Cloud Run'da çalıştırır (private servis). Bu servis artık hem `/models/generate` endpoint'i ile LLM çıkarımı hem de `/embed` endpoint'i ile embedding üretimi yapabilir.

**Beklenen çıktı:** `Service [model-inferencer] revision [model-inferencer-xxxxx] has been deployed and is serving 100 percent of traffic.`

---

### 6. ChromaDB Servisini Dağıt

```cmd
gcloud run deploy chromadb --image chromadb/chroma:latest --platform managed --region %REGION% --no-allow-unauthenticated --cpu=2 --memory=4Gi --port=8000 --project=%PROJECT_ID%
```

**Ne işe yarar:** ChromaDB vector veritabanı servisini Cloud Run'da çalıştırır (private servis). Bu servis kalıcı vector depolama sağlar.

**Beklenen çıktı:** `Service [chromadb] revision [chromadb-xxxxx] has been deployed and is serving 100 percent of traffic.`

---

### 7. API-Gateway İmajını Oluştur

```cmd
copy Dockerfile.api Dockerfile
gcloud builds submit . --tag %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/api-gateway --project=%PROJECT_ID%
del Dockerfile
```

**Ne işe yarar:** Ana API gateway servisinin Docker imajını oluşturur. Önce Dockerfile.api'yi Dockerfile olarak kopyalar, build yapar, sonra geçici Dockerfile'ı siler.

**Beklenen çıktı:** `SUCCESS` mesajı.

---

### 8. PDF-Processor URL'sini Al

```cmd
gcloud run services describe pdf-processor --platform managed --region %REGION% --format "value(status.url)" --project=%PROJECT_ID%
```

**Ne işe yarar:** PDF-Processor servisinin URL'sini getirir.

**⚠️ ÖNEMLİ:** Bu komutun çıktısını kopyalayın! Örnek:

```
https://pdf-processor-abc123-ew.a.run.app
```

**📝 Bu URL'yi not edin - Adım 11'de kullanacaksınız!**

---

### 9. Model-Inferencer URL'sini Al

```cmd
gcloud run services describe model-inferencer --platform managed --region %REGION% --format "value(status.url)" --project=%PROJECT_ID%
```

**Ne işe yarar:** Model-Inferencer servisinin URL'sini getirir.

**⚠️ ÖNEMLİ:** Bu komutun çıktısını da kopyalayın! Örnek:

```
https://model-inferencer-xyz789-ew.a.run.app
```

**📝 Bu URL'yi de not edin - Adım 11'de kullanacaksınız!**

---

### 10. ChromaDB URL'sini Al

```cmd
gcloud run services describe chromadb --platform managed --region %REGION% --format "value(status.url)" --project=%PROJECT_ID%
```

**Ne işe yarar:** ChromaDB servisinin URL'sini getirir.

**⚠️ ÖNEMLİ:** Bu komutun çıktısını da kopyalayın! Örnek:

```
https://chromadb-uvw456-ew.a.run.app
```

**📝 Bu URL'yi de not edin - Adım 11'de kullanacaksınız!**

---

### 11. API-Gateway Servisini URL'lerle Dağıt

**🔥 DİKKAT:** Aşağıdaki komutta `PDF_PROCESSOR_URL_BURAYA`, `MODEL_INFERENCE_URL_BURAYA` ve `CHROMADB_URL_BURAYA` kısımlarını Adım 8, 9 ve 10'da aldığınız **gerçek URL'lerle** değiştirin.

**Örnek Değiştirme:**

- `PDF_PROCESSOR_URL_BURAYA` → `https://pdf-processor-abc123-ew.a.run.app`
- `MODEL_INFERENCE_URL_BURAYA` → `https://model-inferencer-xyz789-ew.a.run.app`
- `CHROMADB_URL_BURAYA` → `https://chromadb-uvw456-ew.a.run.app`

**Değiştirilmiş komut örneği:**

```cmd
gcloud run deploy api-gateway --image %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/api-gateway --platform managed --region %REGION% --allow-unauthenticated --cpu=1 --memory=512Mi --set-env-vars="PDF_PROCESSOR_URL=https://pdf-processor-abc123-ew.a.run.app,MODEL_INFERENCE_URL=https://model-inferencer-xyz789-ew.a.run.app,CHROMADB_URL=https://chromadb-uvw456-ew.a.run.app" --project=%PROJECT_ID%
```

**Sizin kullanacağınız komut:**

```cmd
gcloud run deploy api-gateway --image %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/api-gateway --platform managed --region %REGION% --allow-unauthenticated --cpu=1 --memory=512Mi --set-env-vars="PDF_PROCESSOR_URL=PDF_PROCESSOR_URL_BURAYA,MODEL_INFERENCE_URL=MODEL_INFERENCE_URL_BURAYA,CHROMADB_URL=CHROMADB_URL_BURAYA" --project=%PROJECT_ID%
```

**Ne işe yarar:** Ana API gateway'i çalıştırır ve diğer servislerin URL'lerini ortam değişkeni olarak verir. Bu servis public olacak.

**Beklenen çıktı:** `Service [api-gateway] revision [api-gateway-xxxxx] has been deployed and is serving 100 percent of traffic.`

---

### 12. Ana API URL'sini Al (Son Adım!)

```cmd
gcloud run services describe api-gateway --platform managed --region %REGION% --format "value(status.url)" --project=%PROJECT_ID%
```

**Ne işe yarar:** Ana sisteminizin public URL'sini verir.

**Çıktı örneği:**

```
https://api-gateway-def456-ew.a.run.app
```

## 🎉 Tebrikler!

Dağıtımınız tamamlandı. Yukarıdaki URL'yi kullanarak sisteminize erişebilirsiniz.

---

## Hızlı Test Komutları

Ana URL'nizi aldıktan sonra, çalışıp çalışmadığını test edin:

```cmd
curl https://sizin-api-gateway-url/health
```

```cmd
curl https://sizin-api-gateway-url/models
```

---

## Sorun Giderme

### 🚨 "Container failed to start" Hatası - KRİTİK DÜZELTME

**Problem:** Cloud Run'da "container failed to start" hatası alıyorsanız, bunun nedeni servislerin Cloud Run'ın sağladığı `PORT` ortam değişkenini dinlememesi olabilir.

**Çözüm:** Bu hata için kritik düzeltmeler yapılmıştır:

✅ **pdf-processor** servisi artık `PORT` değişkenini okuyor
✅ **model-inferencer** servisi artık `PORT` değişkenini okuyor
✅ **api-gateway** servisi zaten `PORT` değişkenini okuyor

**⚠️ ÖNEMLİ:** Bu düzeltmelerden sonra **2. Adımdan itibaren** tüm komutları tekrar çalıştırmanız gerekmektedir:

```cmd
gcloud builds submit ./services/pdf_processing_service --tag %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/pdf-processor --project=%PROJECT_ID%
```

```cmd
gcloud builds submit ./services/model_inference_service --tag %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/model-inferencer --project=%PROJECT_ID%
```

```cmd
copy Dockerfile.api Dockerfile
gcloud builds submit . --tag %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/api-gateway --project=%PROJECT_ID%
del Dockerfile
```

Daha sonra **3. Adımdan** devam ederek tüm servisleri yeniden dağıtın.

---

### 🚨 HATA: `pkill: not found` (Build Failure)

**Problem:** Model-inferencer imajını oluştururken (`gcloud builds submit ./services/model_inference_service`) şu hatayı alabilirsiniz:

```
/bin/sh: 1: pkill: not found
```

**Neden:** Minimal Docker imajında (`python:3.9-slim`) `pkill` komutu bulunmaz. `pkill` komutu `procps` paketinde yer alır ancak bu paket varsayılan olarak yüklü değildir.

**Çözüm:** Bu hata bu görevle düzeltilmiştir. `services/model_inference_service/Dockerfile` dosyasında `procps` paketi kurulum aşamasına eklenmiştir.

**⚠️ ÖNEMLİ:** Bu düzeltmeden sonra **4. Adım'dan (`gcloud builds submit` for model-inferencer)** itibaren komutları tekrar çalıştırmanız gerekmektedir:

```cmd
gcloud builds submit ./services/model_inference_service --tag %REGION%-docker.pkg.dev/%PROJECT_ID%/rag-services/model-inferencer --project=%PROJECT_ID%
```

Daha sonra **5. Adım'dan** devam ederek model-inferencer servisini yeniden dağıtın.

---

### Diğer Sorunlar

**"Permission denied" hatası:**

- `gcloud auth login` komutunu çalıştırın

**"Project not found" hatası:**

- `PROJECT_ID` değişkenini kontrol edin

**URL boş gelirse:**

- 1-2 dakika bekleyin, komutları tekrar çalıştırın

**Build timeout:**

- İnternet bağlantınızı kontrol edin

---

Bu rehber tamamen şeffaftır ve her adım açık şekilde belirtilmiştir. Hiçbir gizli işlem yoktur.

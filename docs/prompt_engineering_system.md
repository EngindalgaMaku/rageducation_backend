# 🎨 Öğretmen Prompt Mühendisliği Sistemi

## Genel Bakış

Bu sistem, öğretmenlerin RAG (Retrieval-Augmented Generation) sisteminde prompt mühendisliği yapabilmeleri için tasarlanmış kapsamlı bir araçtır. Öğretmenler bu sistem sayesinde:

- **Özel prompt şablonları** oluşturabilir
- **Hızlı komutlar** tanımlayabilir (/komut-adi formatında)
- **Prompt performansını** izleyebilir ve analiz edebilir
- **A/B testleri** yaparak prompt'ları optimize edebilir
- **Hazır eğitim prompt'larını** kullanabilir

## 🏗️ Sistem Mimarisi

### Ana Bileşenler

1. **TeacherPromptManager** (`src/services/prompt_manager.py`)

   - Prompt ve komut yönetimi
   - SQLite veritabanı ile veri saklama
   - Performans metriklerini kaydetme

2. **Prompt Engineering Panel** (`src/interface/prompt_engineering_panel.py`)

   - Streamlit tabanlı kullanıcı arayüzü
   - 5 ana sekme (Şablonlar, Komutlar, Test, Analytics, Kitaplık)

3. **App Logic Integration** (`src/app_logic.py`)
   - RAG sistemi ile entegrasyon
   - Performans izleme
   - Otomatik etkinlik analizi

## 🚀 Özellikler

### 1. Özel Prompt Şablonları

Öğretmenler kendi özel prompt şablonlarını oluşturabilir:

```python
# Örnek şablon
template = """Sen bir {ders} öğretmenisin. {konu} konusunu {seviye} seviyesinde açıkla.

KONU: {konu}
ÖĞRENCİ SEVİYESİ: {seviye}
AÇIKLAMA STİLİ: {stil}

KURALLAR:
- Basit ve anlaşılır dil kullan
- Örnekler ver
- Adım adım açıkla

AÇIKLAMA:"""
```

**Özellikler:**

- Değişken desteği (`{degisken_adi}`)
- Kategori sistemi (Eğitsel, Ders Özel, Değerlendirme, vb.)
- Karmaşıklık seviyesi (Başlangıç, Orta, İleri, Uzman)
- Etiket sistemi
- Çok dilli destek (TR/EN)

### 2. Hızlı Komut Sistemi

Öğretmenler sık kullandıkları prompt'lar için kısayol komutları oluşturabilir:

#### Varsayılan Komutlar

- **`/basit-anlat`** - Konuyu basit ve anlaşılır şekilde açıkla
- **`/analoji-yap`** - Günlük hayattan örneklerle açıkla
- **`/soru-sor`** - Konuyla ilgili düşündürücü sorular sor
- **`/ozet-cikar`** - Uzun metni önemli noktalarıyla özetle
- **`/test-hazirla`** - Konuyla ilgili test soruları hazırla

#### Komut Kullanımı

```bash
/basit-anlat topic="Fotosentez" grade_level="5. sınıf"
/analoji-yap topic="Atom yapısı" audience="9. sınıf"
/soru-sor topic="Çevre kirliliği" level="7. sınıf" question_type="tartışma"
```

### 3. Performans İzleme ve Analytics

#### Metrikler

- **Çalışma Süresi:** Prompt'un yanıt süresi
- **Kullanıcı Memnuniyeti:** 1-5 arası rating
- **Cevap Kalitesi:** Otomatik ve manuel değerlendirme
- **Eğitsel Verimlilik:** Eğitim açısından etkinlik
- **Öğrenci İlgisi:** Engagement puanı

#### Otomatik Analiz

Sistem her prompt çıktısını otomatik olarak analiz eder:

```python
# Analiz metrikleri
analysis = {
    "response_length": 450,
    "source_count": 3,
    "avg_relevance_score": 0.85,
    "estimated_quality": 0.78,
    "educational_indicators": {
        "explanatory_language": True,
        "structured_content": True,
        "interactive_elements": False
    }
}
```

### 4. Test ve Karşılaştırma

- **Tek Prompt Testi:** Belirli bir prompt'u test et
- **Çoklu Cevap Testi:** Aynı soru için farklı prompt'larla cevap üret
- **A/B Testing:** İki farklı prompt'u karşılaştır
- **Toplu Test:** Birden fazla prompt'u otomatik test et

### 5. Hazır Prompt Kitaplığı

Eğitim alanlarına göre düzenlenmiş hazır prompt'lar:

- **🎓 Genel Eğitim** - Konu açıklama, soru üretme
- **🧮 Matematik** - Problem çözme, adım adım açıklama
- **🔬 Fen Bilimleri** - Deney açıklaması, kavram açıklama
- **📝 Türkçe** - Metin analizi, yaratıcı yazma
- **🌍 Sosyal Bilgiler** - Tarih analizi, coğrafya açıklaması

## 💻 Teknik Detaylar

### Veritabanı Yapısı

```sql
-- Özel Prompt'lar
CREATE TABLE custom_prompts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    template TEXT NOT NULL,
    category TEXT NOT NULL,
    complexity TEXT NOT NULL,
    language TEXT NOT NULL,
    variables TEXT,  -- JSON array
    tags TEXT,       -- JSON array
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    usage_count INTEGER DEFAULT 0,
    avg_rating REAL DEFAULT 0.0,
    is_active BOOLEAN DEFAULT 1
);

-- Prompt Komutları
CREATE TABLE prompt_commands (
    id TEXT PRIMARY KEY,
    command TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    prompt_template TEXT NOT NULL,
    parameters TEXT,    -- JSON array
    examples TEXT,      -- JSON array
    subject_area TEXT,
    grade_level TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    usage_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT 1
);

-- Performans Metrikleri
CREATE TABLE prompt_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT NOT NULL,
    execution_time REAL,
    user_rating REAL,
    response_quality REAL,
    educational_effectiveness REAL,
    engagement_score REAL,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    user_feedback TEXT
);
```

### API ve Fonksiyonlar

#### Prompt Yönetimi

```python
# Özel prompt oluştur
prompt = teacher_prompt_manager.create_custom_prompt(
    name="Matematik Açıklama",
    template="Sen bir matematik öğretmenisin...",
    category=PromptCategory.SUBJECT_SPECIFIC,
    complexity=PromptComplexity.INTERMEDIATE,
    language="tr"
)

# Komut çalıştır
filled_prompt, error = teacher_prompt_manager.execute_prompt_command(
    "/basit-anlat",
    topic="Çarpma işlemi",
    grade_level="3. sınıf"
)

# Performans kaydet
performance = PromptPerformance(
    prompt_id=prompt.id,
    execution_time=2.3,
    user_rating=4.5,
    educational_effectiveness=4.0,
    timestamp=datetime.now().isoformat()
)
teacher_prompt_manager.record_prompt_performance(performance)
```

#### RAG Entegrasyonu

```python
# RAG ile komut çalıştır
answer, sources, scores, metas, performance = execute_prompt_command_with_rag(
    vector_store=store,
    command="/basit-anlat",
    param_values={"topic": "fotosentez", "grade_level": "5. sınıf"},
    generation_model="mistral:7b"
)

# Etkinlik analizi
analysis = analyze_prompt_effectiveness(
    answer=answer,
    sources=sources,
    scores=scores,
    query_params=param_values,
    execution_time=performance["execution_time"]
)
```

## 🎯 Kullanım Senaryoları

### Senaryo 1: Matematik Öğretmeni

Bir matematik öğretmeni 5. sınıf öğrencileri için kesirler konusunu açıklamak istiyor:

1. **Komut Kullanımı:**

   ```
   /basit-anlat topic="kesirler" grade_level="5. sınıf"
   ```

2. **Sistem Çıktısı:**

   - Öğrenci seviyesine uygun açıklama
   - Günlük hayattan örnekler
   - Görsel açıklamalar
   - Adım adım anlatım

3. **Performans İzleme:**
   - Çalışma süresi: 2.1 saniye
   - Öğrenci ilgisi: 4.3/5
   - Eğitsel verimlilik: 4.5/5

### Senaryo 2: Fen Bilgisi Öğretmeni

Bir fen öğretmeni 7. sınıf için fotosentez deneyi hazırlamak istiyor:

1. **Özel Prompt Oluşturma:**

   ```
   template: "Sen bir fen bilgisi öğretmenisin. {deney_adi} deneyini
   {sinif} sınıf için güvenli şekilde açıkla..."
   ```

2. **Test ve Optimizasyon:**
   - A/B test ile farklı açıklama stilleri
   - Öğrenci geri bildirimlerine göre iyileştirme
   - Performans metriklerini izleme

### Senaryo 3: Türkçe Öğretmeni

Bir Türkçe öğretmeni öğrencilerinin yaratıcı yazma becerilerini geliştirmek istiyor:

1. **Komut Kombinasyonu:**

   ```
   /yaratici-yazma tur="hikaye" konu="dostluk" seviye="6. sınıf"
   /soru-sor topic="yaratıcı yazma" level="orta" question_type="yaratıcı"
   ```

2. **Analytics İncelemesi:**
   - En başarılı prompt şablonları
   - Öğrenci engagement oranları
   - Uzun vadeli performans trendleri

## 🔧 Kurulum ve Yapılandırma

### Gereksinimler

```
streamlit>=1.28.0
sqlite3 (Python built-in)
pandas>=1.5.0
plotly>=5.15.0
```

### Kurulum Adımları

1. **Veritabanı İnitialization:**

   ```python
   from src.services.prompt_manager import TeacherPromptManager
   manager = TeacherPromptManager()  # Otomatik olarak DB oluşturur
   ```

2. **Streamlit Arayüzü:**

   ```python
   from src.interface.prompt_engineering_panel import render_prompt_engineering_panel
   render_prompt_engineering_panel()
   ```

3. **Öğretmen Paneli Entegrasyonu:**
   - Ana teacher_ui.py'ye "🎨 Prompt Mühendisliği" butonu eklendi
   - Sidebar'dan hızlı erişim sağlandı

## 📊 Performans Optimizasyonu

### Önbellek Sistemi

- Sık kullanılan prompt'lar önbellekte saklanır
- Veritabanı sorguları optimize edilmiştir
- Toplu işlemler için batch API'ler mevcuttur

### Ölçeklenebilirlik

- SQLite yerine PostgreSQL kullanılabilir
- Redis önbellek eklenebilir
- Mikroservis mimarisine dönüştürülebilir

## 🔮 Gelecek Geliştirmeler

### Planlandı

1. **Gelişmiş Analytics:**

   - Trend analizi
   - Prediktif modelling
   - Karşılaştırmalı raporlama

2. **Collaboration Özellikleri:**

   - Prompt paylaşımı
   - Topluluk kitaplığı
   - Peer review sistemi

3. **Advanced Prompt Engineering:**

   - Chain-of-thought prompting
   - Few-shot learning integration
   - Dynamic prompt adaptation

4. **Integration Genişlemeleri:**
   - LMS entegrasyonu
   - API endpoints
   - Webhook desteği

## 💡 En İyi Uygulamalar

### Prompt Yazma Rehberi

1. **Net ve Spesifik Olun:**

   ```
   ❌ "Bu konuyu açıkla"
   ✅ "Fotosentez sürecini 5. sınıf öğrencileri için basit kelimelerle açıkla"
   ```

2. **Değişkenleri Etkili Kullanın:**

   ```
   template: "Sen bir {ders} öğretmenisin. {konu} konusunu {seviye}
   seviyesinde {stil} şeklinde açıkla."
   ```

3. **Yapılandırılmış Format:**

   ```
   KURALLAR:
   - Basit kelimeler kullan
   - Örnekler ver
   - Adım adım açıkla

   ÇIKIŞ FORMATI:
   1. Ana kavram
   2. Detaylı açıklama
   3. Örnekler
   ```

### Performans İzleme

1. **Düzenli Analiz:**

   - Haftalık performans raporları
   - Trend takibi
   - Benchmark karşılaştırmaları

2. **Öğrenci Geri Bildirimi:**

   - Rating sistemini aktif kullanın
   - Yazılı feedback'i değerlendirin
   - Engagement metriklerini izleyin

3. **Sürekli İyileştirme:**
   - A/B test sonuçlarına göre optimize edin
   - En başarılı pattern'leri belgelerin
   - Başarısız prompt'ları analiz edin

## 📞 Destek ve Dokümantasyon

- **Kod Dokümantasyonu:** Tüm fonksiyonlar docstring ile dokümante edilmiştir
- **Test Coverage:** Kapsamlı test suite mevcuttur
- **API Referansı:** Detaylı API dokümantasyonu ayrıca mevcuttur
- **Troubleshooting Guide:** Yaygın sorunlar ve çözümleri

---

Bu sistem ile öğretmenler, RAG tabanlı eğitim asistanlarını kendi ihtiyaçlarına göre özelleştirebilir ve sürekli olarak iyileştirebilirler. Prompt mühendisliği artık sadece teknik uzmanların değil, her öğretmenin kullanabileceği bir araç haline gelmiştir. 🎓✨

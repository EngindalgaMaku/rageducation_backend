# PDF Markdown Görüntüleme ve Chunk Analizi Sistemi

Bu doküman RAG3 projesine eklenen **PDF Markdown Görüntüleyici ve Chunk Analizi Sistemi**nin kullanım rehberidir.

## 🎯 Sistem Özellikleri

### 📄 PDF Markdown Görüntüleyici

- **Marker Integration**: PDF'leri yüksek kaliteli Markdown'a dönüştürme
- **Gelişmiş Rendering**: Syntax highlighting ve enhanced markdown görüntüleme
- **Dosya Yönetimi**: Markdown'ları otomatik kaydetme ve yönetme
- **Çoklu Görüntüleme Modları**: Render edilmiş, kaynak kod, yan yana görünümler

### 🧩 Chunk Analizi Sistemi

- **Çoklu Strateji Desteği**: Cümle, paragraf ve karakter bazlı chunking
- **Gelişmiş Görselleştirme**: İstatistiksel grafikler ve karşılaştırma araçları
- **Performans Analizi**: Chunk kalitesi ve tutarlılık metrikleri
- **Export Özelliği**: JSON, CSV ve Markdown rapor formatları

### 🔬 Analiz Dashboard'u

- **Detaylı İstatistikler**: Chunk boyutu, dağılım ve kalite analizi
- **3D Görselleştirmeler**: İnteraktif grafikler ve ısı haritaları
- **Karşılaştırmalı Analiz**: Farklı stratejilerin performans karşılaştırması
- **Timeline Analizi**: Chunk sıralama ve akış görselleştirmesi

## 🚀 Sistem Nasıl Kullanılır?

### 1. Öğretmen Paneli Üzerinden Erişim

```python
# Öğretmen paneline giriş yapın
# Gelişmiş Özellikler bölümünden "📄 PDF Analizi" butonuna tıklayın
```

### 2. PDF Yükleme ve İşleme

#### 2.1 Yeni PDF Yükleme

- **Dosya Seçimi**: PDF dosyanızı yükleyin
- **İşleme Seçenekleri**:
  - ✅ Marker kullan (Gelişmiş) - Yüksek kaliteli conversion
  - ✅ Markdown'ı dosyaya kaydet - Otomatik dosya kaydetme
- **Dönüştürme**: "🔄 Markdown'a Dönüştür" butonuna tıklayın

#### 2.2 Mevcut PDF'lerden Seçim

- **Yüklü Dosyalar**: `data/uploads/` klasöründeki PDF'leri listeler
- **Hızlı Görüntüleme**: Daha önce işlenmiş markdown'ları direkt açabilirsiniz

### 3. Görüntüleme ve Analiz Seçenekleri

#### 3.1 📊 Genel Bakış Sekmesi

- **Dosya Metrikleri**: Sayfa, kelime, karakter sayıları
- **İşleme Bilgileri**: Kullanılan yöntem ve kalite bilgisi
- **İçerik Özellikleri**: Görsel, tablo vb. yapısal elementler

#### 3.2 📝 Markdown Görünümü Sekmesi

**Görünüm Modları:**

- **🎨 Render edilmiş**: Standard Streamlit markdown rendering
- **✨ Gelişmiş Rendering**: Syntax highlighting ile enhanced görünüm
- **📝 Ham Markdown**: Kaynak kod görünümü
- **🔄 Yan Yana**: Render ve kaynak kod birlikte

**Özellikler:**

- Code block syntax highlighting (Python, JS, SQL vb.)
- Mermaid diagram desteği
- Enhanced typography ve formatting
- Markdown istatistikleri

#### 3.3 🧩 Temel Chunk Analizi

- **Strateji Seçimi**: Cümle, paragraf veya karakter bazlı
- **İstatistiksel Görüntüleme**: Chunk boyutları, dağılım grafikleri
- **Chunk Browser**: Individual chunk içeriklerini görüntüleme
- **Karşılaştırmalı Analiz**: Farklı stratejilerin yan yana karşılaştırması

#### 3.4 🔬 Gelişmiş Chunk Analizi

- **Kapsamlı Dashboard**: Tüm chunk analizi araçları
- **Çoklu Boyut Testleri**: Farklı chunk size'larla performans testi
- **Kalite Metrikleri**: Tutarlılık, tekrar oranı, kelime zenginliği
- **3D Görselleştirmeler**: İnteraktif scatter plot'lar
- **Export Araçları**: JSON, CSV, Markdown rapor formatları

#### 3.5 🔍 Detaylı Analiz

- **AI Destekli Analiz**: Döküman türü, eğitim seviyesi tespiti
- **İçerik Kategorilendirme**: Otomatik konu ve beceri analizi
- **Yapısal Özellik Tespiti**: Örnek, alıştırma, soru varlığı analizi

## 🛠️ Gelişmiş Özellikler

### Dosya Yönetimi

- **Otomatik Kaydetme**: `data/markdown/` klasörüne metadata ile birlikte
- **Dosya Adı Normalizasyonu**: Güvenli dosya adları oluşturma
- **Metadata Tracking**: JSON formatında işleme bilgileri
- **Bulk Operations**: Toplu dosya işleme desteği

### Chunk Analizi Algoritmaları

#### Sentence Strategy (Önerilen)

```python
# Cümle sınırlarından chunking
# Anlam bütünlüğü koruma
# Eğitim materyalleri için ideal
chunk_text(text, chunk_size=800, overlap=100, strategy="sentence")
```

#### Paragraph Strategy

```python
# Paragraf bazlı chunking
# Yapısal metinler için uygun
# Uzun açıklamalar için ideal
chunk_text(text, chunk_size=1000, overlap=150, strategy="paragraph")
```

#### Character Strategy

```python
# Sabit boyut chunking
# Teknik dokümanlarda tutarlı performans
# Memory management için öngörülebilir
chunk_text(text, chunk_size=600, overlap=80, strategy="char")
```

### Performans Optimizasyonu

- **Lazy Loading**: Modeller sadece gerektiğinde yüklenir
- **Caching**: İşlenmiş sonuçlar cache'lenir
- **Batch Processing**: Toplu işleme desteği
- **Memory Management**: Büyük dosyalar için memory optimizasyonu

## 📊 Kalite Metrikleri

### Chunk Kalite Skorları

- **Tutarlılık**: Chunk boyutlarının standard sapması
- **Kısa Chunk Oranı**: 50 karakterden kısa chunk'ların yüzdesi
- **Kelime Zenginliği**: Benzersiz kelime oranı
- **Tekrar Oranı**: En sık kullanılan kelimelerin dağılımı

### Genel Kalite Skoru Hesaplama

```python
quality_score = (
    (1 - short_chunks_ratio) * 0.4 +
    vocabulary_density * 0.3 +
    (1 - repetition_ratio) * 0.3
) * 100
```

## 🔧 Teknik Detaylar

### Gerekli Kütüphaneler

```python
# Core dependencies
streamlit
pandas
numpy
matplotlib
plotly
seaborn

# PDF Processing
marker  # Gelişmiş PDF -> Markdown conversion
PyPDF2  # Fallback PDF okuma

# Text Processing
nltk  # Natural language processing (opsiyonel)
```

### Dosya Yapısı

```
data/
├── markdown/           # İşlenmiş markdown dosyaları
│   ├── document.md     # Markdown içerik
│   └── document_metadata.json  # İşleme bilgileri
├── uploads/           # Yüklenen PDF dosyaları
└── cache/            # İşleme cache'i
```

### API Yapısı

```python
from src.interface.pdf_markdown_viewer import pdf_markdown_viewer

# PDF işleme
pdf_markdown_viewer._process_pdf_file(
    pdf_path="document.pdf",
    filename="document.pdf",
    save_to_file=True,
    use_marker=True
)

# Markdown rendering
pdf_markdown_viewer._render_enhanced_markdown(markdown_content)

# Chunk analizi
from src.interface.chunk_analysis_dashboard import chunk_analysis_dashboard
chunk_analysis_dashboard.render_chunk_dashboard(text_content, filename)
```

## 🧪 Test ve Demo

### Test Arayüzü Çalıştırma

```python
# Test arayüzüne erişim
from src.interface.test_pdf_markdown_viewer import render_demo_interface
render_demo_interface()
```

### Test Senaryoları

- **📝 Markdown Rendering Testi**: Farklı markdown öğelerinin rendering kalitesi
- **🧩 Chunk Analizi Testi**: Çeşitli strateji ve boyutlarda test
- **⚡ Performans Testi**: Büyük dosyalar ile hız ve memory kullanımı
- **🔗 Entegrasyon Testi**: Sistem bileşenlerinin uyumluluğu

## 💡 En İyi Kullanım Önerileri

### PDF Dönüştürme İçin

1. **Marker'ı tercih edin**: Daha yüksek kaliteli sonuçlar
2. **Dosyaya kaydetmeyi aktif tutun**: Tekrar işleme gereksinimini azaltır
3. **Büyük dosyalar için sabırlı olun**: İşleme süresi dosya boyutu ile artar

### Chunk Analizi İçin

1. **Sentence stratejisini deneyin**: Çoğu durumda en iyi sonuçları verir
2. **Farklı boyutları karşılaştırın**: İçerik türüne göre optimal boyut değişir
3. **Kalite metriklerini takip edin**: Düşük skorlu chunk'ları inceleyin

### Performans İyileştirme

1. **Cache'i temizlemeyin**: Tekrar işleme süresini kısaltır
2. **Batch işleme yapın**: Çok sayıda dosya için daha verimli
3. **Test modunu kullanın**: Yeni özellikler denerken

## 🚨 Bilinen Limitasyonlar

1. **Marker Dependency**: Marker kurulumu gerekli, yoksa fallback kullanılır
2. **Memory Usage**: Çok büyük PDF'ler memory kullanımını artırabilir
3. **Processing Time**: Detaylı analiz zaman alabilir
4. **Language Support**: Türkçe ve İngilizce optimize edilmiş

## 🔄 Gelecek Geliştirmeler

- [ ] OCR desteği (taranmış PDF'ler için)
- [ ] Toplu PDF işleme arayüzü
- [ ] Custom chunking stratejileri
- [ ] API endpoint'leri
- [ ] Machine learning tabanlı chunk optimizasyonu
- [ ] Collaborative annotation özelliği

## 📞 Destek ve Katkı

Bu sistem RAG3 projesinin bir parçasıdır. Sorunlar veya öneriler için:

- Issue açabilirsiniz
- Pull request gönderebilirsiniz
- Test sonuçlarınızı paylaşabilirsiniz

---

_Bu döküman sistemi en verimli şekilde kullanmanız için hazırlanmıştır. Güncellemeler için düzenli olarak kontrol edin._

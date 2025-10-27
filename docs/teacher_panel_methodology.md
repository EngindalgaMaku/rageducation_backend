# Öğretmen Paneli - Sistem Yönetimi ve Test Arayüzü

## Genel Bakış

Öğretmen paneli, RAG (Retrieval-Augmented Generation) destekli eğitim sisteminin yönetim ve test arayüzüdür. Bu panel, eğitmenlerin ders materyallerini sisteme yüklemesine, işlemesine ve sistemin performansını test etmesine olanak sağlar.

## Ana Bileşenler

### 1. Ders Oturumu Yönetimi (Session Management)

Sistem, ders-bazlı oturum yönetimi yaklaşımı benimser:

- **Oturum Oluşturma**: Her ders için benzersiz oturum adları ile ayrı vektör veritabanları oluşturulur
- **Oturum Geçişi**: Farklı dersler arasında hızlı geçiş imkanı
- **Oturum Durumu**: Mevcut oturumdaki metin parçası sayısı ve vektör boyutu bilgisi
- **Oturum Temizleme**: Mevcut oturumu sıfırlama ve yeni başlama özelliği

**Teknik Detay**: Oturum verileri `data/vector_db/sessions/` dizininde saklanır ve her oturum için ayrı `.index`, `.chunks`, ve `.meta.jsonl` dosyaları oluşturulur.

### 2. Belge Yükleme ve İşleme (Document Processing)

#### 2.1 Desteklenen Dosya Formatları

- **PDF**: PyPDF2 kütüphanesi ile işlenir
- **DOCX**: python-docx kütüphanesi ile işlenir
- **PPTX**: python-pptx kütüphanesi ile işlenir

#### 2.2 Metin Parçalama (Text Chunking) Stratejileri

Sistem üç farklı parçalama stratejisi sunar:

- **Karakter Bazlı (char)**: Sabit karakter sayısına göre parçalama
- **Paragraf Bazlı (paragraph)**: Paragraf sınırlarını koruyarak parçalama
- **Cümle Bazlı (sentence)**: Cümle sınırlarını koruyarak parçalama

#### 2.3 Yapılandırılabilir Parametreler

- **Chunk Boyutu**: 200-2000 karakter arası ayarlanabilir (varsayılan: config'den alınır)
- **Chunk Örtüşmesi**: 0-500 karakter arası örtüşme (varsayılan: config'den alınır)
- **Embedding Modeli**: mxbai-embed-large, nomic-embed-text, snowflake-arctic-embed seçenekleri

#### 2.4 Metadata Oluşturma

Her metin parçası için otomatik metadata oluşturulur:

- `source_file`: Kaynak dosya adı
- `page_number`: PDF sayfa numarası (varsa)
- `slide_number`: PPTX slayt numarası (varsa)
- `title`: Metin parçasının başlığı veya ilk satırı

### 3. RAG Sistemi Test Etme

#### 3.1 Tek Cevap Modu

Standart RAG süreci:

1. **Query Embedding**: Sorunun vektör temsiline çevrilmesi
2. **Similarity Search**: FAISS vektör veritabanında benzerlik araması
3. **Context Building**: Bulunan metin parçalarından bağlam oluşturma
4. **LLM Generation**: Ollama LLM ile cevap üretimi

**Yapılandırılabilir Parametreler:**

- **Kaynak Sayısı (top_k)**: 1-20 arası ayarlanabilir
- **LLM ile Sıralama**: Bulunan kaynakların LLM ile yeniden sıralanması
- **Minimum Benzerlik Skoru**: Düşük skorlu sonuçların filtrelenmesi (0.0-1.0)
- **Maksimum Bağlam Uzunluğu**: LLM'e gönderilecek maksimum karakter sayısı (1000-20000)

#### 3.2 Çoklu Cevap Modu

Aynı soru için farklı perspektiflerden cevap üretme:

- **3 Farklı Cevap**: Her cevap farklı kaynak kombinasyonları kullanır
- **Farklı Temperature Değerleri**: Çeşitlilik için 0.3, 0.5, 0.7 sıcaklık değerleri
- **Kaynak Çeşitliliği**: Her cevap için farklı başlangıç noktalarından kaynak seçimi

### 4. Sonuç Görüntüleme ve Analiz

#### 4.1 Cevap Görüntüleme

- **Syntax Highlighting**: Kod blokları otomatik olarak vurgulanır
- **Markdown Desteği**: Formatlı metin görüntüleme
- **Gerçek Zamanlı Gösterim**: Streamlit ile interaktif arayüz

#### 4.2 Kaynak Detayları

Her kaynak için ayrıntılı bilgi gösterimi:

- **Dosya Bilgisi**: Kaynak dosya adı ve sayfa/slayt numarası
- **Benzerlik Skoru**: Cosine similarity skoru (0-1 arası)
- **Tam Metin İçeriği**: Kullanılan metin parçasının tam içeriği
- **Arama Kılavuzu**: Orijinal dokümanda bulma için Ctrl+F önerisi

#### 4.3 Performans Metrikleri

- **Bulunan Kaynak Sayısı**: Threshold üstü kaynak adedi
- **En Yüksek Skor**: En iyi eşleşme skoru
- **Ortalama Skor**: Tüm kaynakların ortalama benzerlik skoru

### 5. Oturum Yönetimi ve Veri İşlemleri

#### 5.1 İşleme Raporları

Her belge yükleme işlemi sonrası detaylı rapor:

- **İşlenen Dosya Sayısı**: Başarıyla işlenen belge adedi
- **Eklenen Vektör Sayısı**: Vektör veritabanına eklenen yeni girdi sayısı
- **Üretilen Chunk Sayısı**: Oluşturulan metin parçası adedi
- **Kullanılan Strateji**: Seçilen parçalama stratejisi

#### 5.2 Veri Persistency

- **Otomatik Kaydetme**: Her işlem sonrası otomatik veritabanı kaydı
- **Oturum Bazlı Saklama**: Her ders için ayrı vektör veritabanı dosyaları
- **Metadata Saklama**: JSON Lines formatında metadata saklanması

#### 5.3 Hata Yönetimi

- **Belge İşleme Hataları**: Desteklenmeyen format veya bozuk dosya uyarıları
- **LLM Bağlantı Hataları**: Ollama servis bağlantı problemleri için fallback
- **Vektör Veritabanı Hataları**: FAISS index problemleri için uyarı mesajları

## Teknik Mimarı Entegrasyonu

### Vektör Veritabanı (FAISS)

- **Index Tipi**: FAISS IP (Inner Product) index
- **Embedding Boyutu**: Seçilen modele göre değişkenlik
- **Similarity Metric**: Cosine similarity

### LLM Entegrasyonu (Ollama)

- **Model Yönetimi**: Dinamik model seçimi ve konfigürasyonu
- **Prompt Engineering**: Dil-aware prompt template sistemi
- **Response Handling**: Structured response parsing

### Dosya Sistemi Organizasyonu

```
data/
├── uploads/           # Geçici dosya depolama
├── vector_db/
│   └── sessions/      # Oturum bazlı vektör veritabanları
└── cache/            # Embedding ve işleme cache'i
```

## Kullanıcı Deneyimi (UX) Tasarımı

### Adım-Adım İş Akışı

1. **Oturum Seçimi**: Ders adı belirleme ve oturum oluşturma
2. **Belge Yükleme**: Drag-drop veya dosya seçici ile yükleme
3. **Parametre Ayarlama**: İsteğe bağlı gelişmiş ayarlar
4. **İşleme ve Test**: Belge işleme ve sistem testi
5. **Sonuç Analizi**: Detaylı performans ve kaynak analizi

### Görsel Tasarım Prensipleri

- **Konteyner Bazlı Layout**: Her işlev grup için ayrı bordered container
- **Renk Kodlaması**: Başarı (yeşil), uyarı (sarı), hata (kırmızı) durumları
- **Progresif Bilgilendirme**: Detaylar expandable section'larda
- **Responsive Tasarım**: Farklı ekran boyutları için uyarlanabilir düzen

Bu sistem, RAG teknolojisinin eğitim ortamında pratik uygulanması için kapsamlı bir yönetim arayüzü sağlamaktadır.

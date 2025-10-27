# RAG Pipeline Mimarisi

Aşağıda, sistemin bir kullanıcı sorusuna nasıl cevap ürettiğini adım adım gösteren RAG (Retrieval-Augmented Generation) pipeline diyagramı bulunmaktadır. Bu diyagram `Mermaid` formatında oluşturulmuştur.

## Mermaid Kodu

```mermaid
graph TD
    subgraph "1. Sorgu Ön İşleme"
        A[Kullanıcı Sorusu] --> B{Dil Tespiti};
        B --> C[Sorgu Gömme (Query Embedding)];
    end

    subgraph "2. Bilgi Getirme (Retrieval)"
        C --> D[Vektör Veritabanında Arama (FAISS)];
        D --> E{Benzerlik Skoruna Göre Filtrele};
    end

    subgraph "3. Sıralama ve Bağlam Oluşturma"
        E --> F{"LLM ile Yeniden Sıralama (Reranking)<br/><i>(Opsiyonel)</i>"};
        F --> G[Nihai Bağlamı Oluşturma (Context Building)];
    end

    subgraph "4. Cevap Üretimi (Generation)"
        G -- Bağlam --> H[Cevap Üretimi (LLM)];
        A -- Soru --> H;
        H --> I[Oluşturulan Cevap];
    end

    %% Styling
    style F fill:#fff9c4,stroke:#f57f17,stroke-width:2px,stroke-dasharray: 5 5
    style H fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
```

## Diyagramın Açıklaması

1.  **Sorgu Ön İşleme**:

    - Kullanıcının girdiği metin (`Kullanıcı Sorusu`) alınır.
    - Sorunun dili tespit edilir (`Dil Tespiti`).
    - Soru, bir embedding modeli kullanılarak vektör temsiline dönüştürülür (`Sorgu Gömme`).

2.  **Bilgi Getirme (Retrieval)**:

    - Oluşturulan sorgu vektörü, `FAISS` vektör veritabanında en benzer belgeleri (chunk'ları) bulmak için kullanılır.
    - Bulunan sonuçlar, önceden belirlenmiş bir minimum benzerlik skoruna göre filtrelenir.

3.  **Sıralama ve Bağlam Oluşturma**:

    - (Opsiyonel) Filtrelenmiş belgeler, alaka düzeyini artırmak için bir Dil Modeli (LLM) tarafından yeniden sıralanır (`LLM ile Yeniden Sıralama`). Bu adım, en alakalı bilgilerin en üste çıkmasını sağlar.
    - En alakalı belgeler birleştirilerek LLM'e gönderilecek nihai bağlam (`Context`) oluşturulur.

4.  **Cevap Üretimi (Generation)**:
    - Orijinal kullanıcı sorusu ve oluşturulan bağlam, bir prompt şablonu kullanılarak bir araya getirilir.
    - Bu birleşik prompt, nihai cevabı üretmesi için LLM'e gönderilir.
    - LLM tarafından üretilen cevap kullanıcıya sunulur.

## Nasıl Kullanılır?

- **Online Editörler**: Bu kodu [Mermaid Live Editor](https://mermaid.live) gibi online araçlara yapıştırarak diyagramı anında görebilir ve PNG/SVG olarak indirebilirsiniz.
- **VS Code**: `Markdown Mermaid` gibi bir eklenti kurarak bu `.md` dosyasını açtığınızda diyagramı doğrudan görebilirsiniz.
- **Streamlit Arayüzü**: İsterseniz bu diyagramı `st.mermaid(diyagram_kodu)` komutuyla doğrudan web arayüzünüze de ekleyebiliriz.

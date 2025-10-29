# RAG3 Mikroservis Mimarisi Önerisi

## Genel Bakış

Mevcut monolitik backend yapısını 4 ana mikroservise ayırarak daha modüler, ölçeklenebilir ve bakım yapılabilir bir mimari öneriyoruz.

## Mikroservis Bölümlemesi

### 🔧 1. PDF Processing Service (Marker Tabanlı)

**Port:** 8001  
**Sorumluluklar:**

- PDF dosyalarını yüksek kaliteli Markdown'a dönüştürme
- Marker kütüphanesi entegrasyonu ve optimizasyonu
- Büyük PDF'ler için bellek yönetimi
- Model cache yönetimi (Marker modelleri için)
- Fallback PDF işleme (PyPDF2)

**Ana Özellikler:**

- Async PDF processing
- Memory-safe operations (4GB+ PDF desteği)
- Progress tracking
- Error recovery
- Cache-optimized model loading

---

### 🤖 2. Model Inference Service (Grok/Ollama)

**Port:** 8002  
**Sorumluluklar:**

- LLM model çıkarımları (Grok, Ollama)
- Cloud LLM client yönetimi
- Model selection ve switching
- Embedding generation
- Response generation

**Ana Özellikler:**

- Multi-provider support (Groq, Ollama)
- Dynamic model switching
- Request queuing
- Timeout management
- Provider fallback mechanisms

---

### 🌐 3. API Gateway Service

**Port:** 8000  
**Sorumluluklar:**

- Request routing ve koordinasyonu
- Session lifecycle yönetimi
- Authentication/Authorization
- Rate limiting
- API versioning
- Frontend ile iletişim

**Ana Özellikler:**

- RESTful API endpoints
- Session-based operations
- Multi-format document upload
- Query handling
- Response aggregation

---

### 📄 4. Document Processing Service

**Port:** 8003  
**Sorumluluklar:**

- Text chunking (semantic, markdown, hybrid)
- Vector embedding storage (FAISS + ChromaDB)
- Multi-backend vector search operations
- Document metadata yönetimi
- Persistent vector database yönetimi
- Cache yönetimi

**Ana Özellikler:**

- Advanced chunking strategies
- Multi-backend vector similarity search (FAISS + ChromaDB)
- Persistent vector storage with ChromaDB
- Advanced metadata filtering and indexing
- Embedding cache
- Cross-backend search optimization
- Collection management and versioning

## Servis Detayları

### 1. PDF Processing Service

#### Dahil Edilecek Dosyalar:

```
src/document_processing/
├── enhanced_pdf_processor.py (1078+ satır) ⭐ ANA DOSYA
├── pdf_processor.py (fallback)
├── document_processor.py (dispatcher)
├── docx_processor.py
└── pptx_processor.py

src/utils/
├── memory_manager.py ⭐ KRITIK
├── model_cache_manager.py ⭐ KRITIK
├── helpers.py
└── logger.py
```

#### API Endpoints:

- `POST /convert/pdf-to-markdown`
- `GET /convert/status/{job_id}`
- `GET /health`
- `GET /models/status`

#### Bağımlılıklar:

- marker-pdf
- PyPDF2
- psutil (memory monitoring)
- threading/concurrent.futures

---

### 2. Model Inference Service

#### Dahil Edilecek Dosyalar:

```
src/utils/
├── cloud_llm_client.py ⭐ ANA DOSYA
├── model_selector.py ⭐ ANA DOSYA
└── prompt_templates.py

src/embedding/
└── embedding_generator.py ⭐ KRITIK

src/rag/
├── rag_pipeline.py (generation kısmı)
└── re_ranker.py

İlgili config ve utility dosyaları
```

#### API Endpoints:

- `POST /models/generate`
- `POST /models/embed`
- `GET /models/available`
- `POST /models/select`
- `GET /health`

#### Bağımlılıklar:

- ollama (opsiyonel)
- requests (Groq API için)
- sentence-transformers
- numpy

---

### 3. API Gateway Service

#### Dahil Edilecek Dosyalar:

```
src/api/
├── main.py ⭐ ANA DOSYA (1177 satır)
├── feedback_api.py
└── main_minimal.py

src/
├── api_server.py
└── app_logic.py ⭐ KOORDINASYON

src/services/
├── session_manager.py ⭐ KRITIK
├── learning_loop_manager.py
└── feedback_processor.py

src/config.py
```

#### API Endpoints:

- `GET /` (health check)
- `POST /sessions`
- `GET /sessions`
- `POST /documents/upload`
- `POST /rag/query`
- `POST /rag/configure-and-process`
- Tüm frontend API endpoints

#### Bağımlılıklar:

- FastAPI
- SQLite (session management)
- Pydantic models

---

### 4. Document Processing Service

#### Dahil Edilecek Dosyalar:

```
src/text_processing/ (TÜMÜ)
├── semantic_chunker.py ⭐ KRITIK
├── text_chunker.py ⭐ KRITIK
├── adaptive_chunk_refiner.py
└── advanced_chunk_validator.py

src/vector_store/
├── faiss_store.py ⭐ KRITIK
├── chroma_store.py ⭐ KRITIK (YENİ)
└── vector_store_manager.py ⭐ YENİ (Multi-backend)

src/embedding/
└── embedding_generator.py (kopya/paylaşımlı)

src/utils/
├── cache.py
├── language_detector.py
└── performance_monitor.py
```

#### API Endpoints:

- `POST /process/chunk`
- `POST /vector/store`
- `POST /vector/search`
- `GET /vector/stats`
- `POST /collections/create`
- `GET /collections/list`
- `DELETE /collections/{collection_name}`
- `GET /collections/{collection_name}/stats`
- `POST /vector/migrate` (FAISS ↔ ChromaDB)
- `GET /health`

#### Bağımlılıklar:

- FAISS (hızlı in-memory search)
- ChromaDB (persistent vector database)
- numpy
- sentence-transformers
- regex
- sqlite3 (ChromaDB backend)

---

### 🗂️ 5. ChromaDB Vector Database Service

**Port:** 8004
**Sorumluluklar:**

- Persistent vector storage ve retrieval
- Collection management ve versioning
- Metadata filtering ve complex queries
- Backup ve restore operations
- Multi-tenant collection isolation
- Vector similarity search optimizations

**Ana Özellikler:**

- HTTP API tabanlı vector operations
- Scalable persistent storage
- Advanced metadata filtering
- Collection-based organization
- Built-in embedding functions
- REST API with OpenAPI documentation

#### API Endpoints:

- `POST /api/v1/collections`
- `GET /api/v1/collections`
- `POST /api/v1/collections/{collection_name}/add`
- `POST /api/v1/collections/{collection_name}/query`
- `GET /api/v1/collections/{collection_name}`
- `DELETE /api/v1/collections/{collection_name}`

#### Bağımlılıklar:

- ChromaDB server
- SQLite (metadata storage)
- Docker runtime
- HTTP client libraries

## Vector Storage Strategy

### Hybrid Approach: FAISS + ChromaDB

**FAISS (Fast Retrieval):**

- In-memory vector search (milisaniye response)
- Büyük dataset'lerde ultra-hızlı similarity search
- RAM-based operations
- Session bazlı temporary storage

**ChromaDB (Persistent Storage):**

- Kalıcı vector storage ve metadata
- Collection-based organization
- Advanced filtering capabilities
- Session ve document versioning
- Backup ve restore operations

### Storage Decision Logic:

```python
def choose_vector_backend(query_type, data_size, persistence_required):
    if persistence_required and data_size < 1M_vectors:
        return "chromadb"
    elif query_type == "realtime" and data_size < 100K_vectors:
        return "faiss"
    else:
        return "hybrid"  # FAISS for speed + ChromaDB for persistence
```

## Servisler Arası İletişim

### Communication Patterns:

```mermaid
graph TB
    F[Frontend] --> AG[API Gateway :8000]

    AG --> PDF[PDF Processing :8001]
    AG --> MI[Model Inference :8002]
    AG --> DP[Document Processing :8003]

    PDF --> DP
    DP --> MI
    DP --> CDB[ChromaDB :8004]

    subgraph "Vector Storage Layer"
        DP -.-> FAISS[(FAISS Store)]
        DP -.-> VManager[(Vector Store Manager)]
        VManager --> CDB
        VManager --> FAISS
    end

    subgraph "İç Servislerde"
        PDF -.-> Cache[(Model Cache)]
        CDB -.-> Storage[(Persistent Storage)]
        MI -.-> LLM[(LLM Models)]
    end
```

### Tipik Request Flow:

1. **Document Upload & Storage:**

   ```
   Frontend → API Gateway → PDF Processing → Document Processing →
   Vector Store Manager → [FAISS + ChromaDB] → Persistent Storage
   ```

2. **RAG Query (Hybrid Search):**

   ```
   Frontend → API Gateway → Document Processing → Vector Store Manager →
   [FAISS (fast) + ChromaDB (metadata filtering)] → Model Inference → Response
   ```

3. **Session Management:**

   ```
   Frontend → API Gateway → Session Manager → Database + ChromaDB Collections
   ```

4. **Collection Management:**

   ```
   Frontend → API Gateway → Document Processing → ChromaDB Service →
   Collection Operations (Create/List/Delete)
   ```

5. **Vector Migration (FAISS ↔ ChromaDB):**
   ```
   Admin → API Gateway → Document Processing → Vector Store Manager →
   [Export from Source] → [Import to Target] → Validation
   ```

## Deployment Yapılandırması

### Docker Compose Örneği:

```yaml
services:
  api-gateway:
    build: ./services/api-gateway
    ports:
      - "8000:8000"
    depends_on:
      - pdf-processing
      - model-inference
      - document-processing
      - chromadb

  pdf-processing:
    build: ./services/pdf-processing
    ports:
      - "8001:8001"
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache

  model-inference:
    build: ./services/model-inference
    ports:
      - "8002:8002"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}

  document-processing:
    build: ./services/document-processing
    ports:
      - "8003:8003"
    volumes:
      - ./data/vector_db:/app/vector_db
      - ./data/chromadb:/app/chromadb
    environment:
      - CHROMADB_HOST=chromadb
      - CHROMADB_PORT=8000
    depends_on:
      - chromadb

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8004:8000"
    volumes:
      - ./data/chromadb:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
```

## Avantajlar

### ✅ Modülerlik

- Her servis bağımsız geliştirilebilir
- Farklı teknolojiler/diller kullanılabilir
- Takım üyeleri specialization yapabilir

### ✅ Ölçeklenebilirlik

- PDF processing CPU-intensive → Daha fazla kaynak
- Model inference GPU-intensive → GPU instance'lar
- API Gateway load balancing → Horizontal scaling

### ✅ Hataya Dayanıklılık

- Bir servis çökerse diğerleri çalışmaya devam eder
- Circuit breaker patterns
- Graceful degradation

### ✅ Deployment Esnekliği

- Servisler bağımsız deploy edilebilir
- Rolling updates
- A/B testing per service

## Geçiş Stratejisi

### Faz 1: Servis Ayırma

1. PDF Processing Service'i çıkar
2. Model Inference Service'i ayır
3. API Gateway'i refactor et
4. Document Processing Service'i oluştur

### Faz 2: İletişim Kurma

1. HTTP REST API'leri tanımla
2. Error handling ve retry logic
3. Health checks ve monitoring

### Faz 3: Optimizasyon

1. Caching strategies
2. Connection pooling
3. Performance monitoring

## Teknisk Gereksinimler

### Shared Dependencies:

- Python 3.9+
- FastAPI
- Pydantic
- Docker & Docker Compose

### Service-Specific:

- **PDF Processing:** marker-pdf, PyPDF2, psutil
- **Model Inference:** ollama, groq, sentence-transformers
- **API Gateway:** SQLite, authentication libs
- **Document Processing:** FAISS, numpy, regex, ChromaDB client
- **ChromaDB Service:** ChromaDB server, Docker, SQLite backend

## ChromaDB Integration Benefits

### ✅ Persistent Vector Storage

- Veriler sistem yeniden başlatıldığında korunur
- Session ve document versioning
- Metadata ile complex filtering
- Collection-based organization

### ✅ Scalability & Performance

- Büyük vector datasets için optimized
- Built-in indexing ve caching
- HTTP API ile service separation
- Multi-tenant support

### ✅ Advanced Features

- Embedding functions support
- Metadata filtering ve search
- Backup ve restore capabilities
- OpenAPI documentation

### ✅ Development & Operations

- Docker-based deployment
- RESTful API interface
- Built-in monitoring ve logging
- Easy integration with existing services

## Migration Strategy

### Phase 1: ChromaDB Integration

1. 🔄 ChromaDB service setup
2. 🔄 Vector Store Manager implementation
3. 🔄 FAISS + ChromaDB hybrid logic
4. 🔄 Collection management APIs

### Phase 2: Data Migration

1. 🔄 Existing FAISS data export
2. 🔄 ChromaDB collection creation
3. 🔄 Vector migration tools
4. 🔄 Validation ve testing

### Phase 3: Production Deployment

1. 🔄 Docker compose updates
2. 🔄 Service coordination
3. 🔄 Performance testing
4. 🔄 Monitoring ve logging

## Sonraki Adımlar

1. ✅ Mimari tasarımı onayı (ChromaDB dahil)
2. 🔄 ChromaDB service setup ve testing
3. 🔄 Vector Store Manager implementation
4. 🔄 Her servis için ayrı klasör yapısı oluşturma
5. 🔄 API interface tanımları (ChromaDB endpoints dahil)
6. 🔄 Docker configuration'ları (ChromaDB service dahil)
7. 🔄 İlk servis (PDF Processing) ayırma
8. 🔄 FAISS → ChromaDB migration tools
9. 🔄 Test ve integration (hybrid vector storage)

---

Bu mimari, kullanıcının belirttiği gereksinimleri karşılar ve mevcut kodun büyük kısmını koruyarak güvenli bir geçiş sağlar.

---

## Data Processing Pipeline Şeması

### 📊 Genel Pipeline Akışı

```mermaid
flowchart TD
    %% Input Layer
    UI[User Interface] --> AG[API Gateway :8000]

    %% Document Upload Pipeline
    AG --> |1. Document Upload| DU{Document Type?}
    DU --> |PDF| PDF[PDF Processing :8001]
    DU --> |DOCX/PPTX| DOC[Document Processing :8003]

    %% PDF Processing Pipeline
    PDF --> |2. Marker Conversion| MARK[Marker PDF→MD]
    MARK --> |3. Memory Management| MEM[Memory Manager]
    MEM --> |4. Processed Markdown| DP[Document Processing :8003]

    %% Document Processing Pipeline
    DP --> |5. Text Analysis| CHUNK{Chunking Strategy}
    CHUNK --> |Semantic| SC[Semantic Chunker]
    CHUNK --> |Markdown| MC[Markdown Chunker]
    CHUNK --> |Hybrid| HC[Hybrid Chunker]

    SC --> EMBED[Embedding Generation]
    MC --> EMBED
    HC --> EMBED

    %% Vector Storage Pipeline
    EMBED --> |6. Vector Storage| VS{Storage Strategy}
    VS --> |Fast Access| FAISS[FAISS Store]
    VS --> |Persistent| CDB[ChromaDB :8004]
    VS --> |Hybrid| BOTH[FAISS + ChromaDB]

    %% Storage Persistence
    FAISS --> |Session Based| TEMP[(Temporary Storage)]
    CDB --> |Collections| PERSIST[(Persistent Storage)]
    BOTH --> TEMP
    BOTH --> PERSIST

    %% Query Pipeline
    AG --> |7. RAG Query| QP[Query Processing]
    QP --> |8. Vector Search| VSM[Vector Store Manager]
    VSM --> |9. Similarity Search| SEARCH{Search Strategy}

    SEARCH --> |Real-time| FAISS
    SEARCH --> |Complex Query| CDB
    SEARCH --> |Hybrid Search| BOTH

    %% Response Generation Pipeline
    SEARCH --> |10. Retrieved Context| MI[Model Inference :8002]
    MI --> |11. LLM Processing| LLM{Model Selection}
    LLM --> |Groq| GROQ[Groq API]
    LLM --> |Ollama| OLLAMA[Ollama Local]

    GROQ --> |12. Generated Response| AG
    OLLAMA --> |12. Generated Response| AG
    AG --> |13. Final Response| UI

    %% Background Processes
    DP -.-> |Session Management| SM[Session Manager]
    SM -.-> |Metadata| DB[(SQLite DB)]

    style PDF fill:#e1f5fe
    style DP fill:#f3e5f5
    style MI fill:#e8f5e8
    style CDB fill:#fff3e0
    style AG fill:#fce4ec
```

### 🔄 Pipeline States ve Transitions

#### 1. Document Ingestion State Machine

```mermaid
stateDiagram-v2
    [*] --> Uploaded: Document Upload
    Uploaded --> ValidatingFormat: Format Check
    ValidatingFormat --> ProcessingPDF: PDF Detected
    ValidatingFormat --> ProcessingDOC: DOCX/PPTX Detected
    ValidatingFormat --> ErrorInvalidFormat: Unknown Format

    ProcessingPDF --> MarkerConversion: Marker Processing
    MarkerConversion --> MemoryOptimization: Large File Check
    MemoryOptimization --> ChunkingReady: Memory Managed
    MarkerConversion --> ChunkingReady: Small File

    ProcessingDOC --> ChunkingReady: Direct Processing

    ChunkingReady --> SemanticChunking: Strategy: Semantic
    ChunkingReady --> MarkdownChunking: Strategy: Markdown
    ChunkingReady --> HybridChunking: Strategy: Hybrid

    SemanticChunking --> EmbeddingGeneration
    MarkdownChunking --> EmbeddingGeneration
    HybridChunking --> EmbeddingGeneration

    EmbeddingGeneration --> VectorStorage: Embeddings Ready
    VectorStorage --> ProcessingComplete: Storage Success
    VectorStorage --> ErrorStorage: Storage Failed

    ProcessingComplete --> [*]
    ErrorInvalidFormat --> [*]
    ErrorStorage --> [*]
```

#### 2. Query Processing State Machine

```mermaid
stateDiagram-v2
    [*] --> QueryReceived: User Query
    QueryReceived --> QueryAnalysis: Analyze Intent
    QueryAnalysis --> VectorSearch: Search Strategy Selected

    VectorSearch --> FAISSSearch: Fast Retrieval Needed
    VectorSearch --> ChromaSearch: Complex Query
    VectorSearch --> HybridSearch: Optimal Strategy

    FAISSSearch --> ContextRetrieved: Results Found
    ChromaSearch --> ContextRetrieved: Results Found
    HybridSearch --> ContextRetrieved: Results Found

    VectorSearch --> NoResults: No Matches
    NoResults --> FallbackSearch: Try Alternative
    FallbackSearch --> ContextRetrieved: Fallback Success
    FallbackSearch --> EmptyResponse: No Context Available

    ContextRetrieved --> ModelInference: Context Ready
    ModelInference --> GroqProcessing: Groq Selected
    ModelInference --> OllamaProcessing: Ollama Selected

    GroqProcessing --> ResponseGenerated: Groq Response
    OllamaProcessing --> ResponseGenerated: Ollama Response
    ModelInference --> ModelError: Inference Failed

    ResponseGenerated --> [*]
    EmptyResponse --> [*]
    ModelError --> [*]
```

### 🏗️ Service Communication Pipeline

#### Inter-Service Data Flow

```mermaid
sequenceDiagram
    participant U as User/Frontend
    participant AG as API Gateway
    participant PDF as PDF Service
    participant DP as Doc Processing
    participant VS as Vector Store Manager
    participant CDB as ChromaDB
    participant FAISS as FAISS Store
    participant MI as Model Inference

    %% Document Upload Flow
    U->>AG: POST /documents/upload
    AG->>PDF: POST /convert/pdf-to-markdown

    Note over PDF: Marker Processing<br/>Memory Management<br/>Cache Optimization

    PDF-->>AG: Processed Markdown
    AG->>DP: POST /process/chunk

    Note over DP: Chunking Strategy<br/>Text Analysis<br/>Embedding Generation

    DP->>VS: Vector Storage Request

    par Parallel Storage
        VS->>FAISS: Store in Memory
        VS->>CDB: POST /api/v1/collections/{id}/add
    end

    VS-->>DP: Storage Confirmation
    DP-->>AG: Processing Complete
    AG-->>U: Upload Success

    %% Query Processing Flow
    U->>AG: POST /rag/query
    AG->>DP: Query Analysis
    DP->>VS: Vector Search Request

    alt Fast Query
        VS->>FAISS: Similarity Search
    else Complex Query
        VS->>CDB: POST /api/v1/collections/{id}/query
    else Hybrid Search
        par
            VS->>FAISS: Memory Search
        and
            VS->>CDB: Persistent Search
        end
        Note over VS: Result Fusion & Ranking
    end

    VS-->>DP: Retrieved Context
    DP->>MI: Generate Response

    alt Groq Model
        MI->>MI: Groq API Call
    else Ollama Model
        MI->>MI: Local Model Inference
    end

    MI-->>DP: Generated Response
    DP-->>AG: Final Response
    AG-->>U: Query Result
```

### 📈 Pipeline Performance Metrics

#### Processing Stages Monitoring

```mermaid
gantt
    title RAG Pipeline Performance Timeline
    dateFormat X
    axisFormat %s

    section PDF Processing
    Document Upload    :milestone, upload, 0, 0
    Marker Conversion  :active, marker, after upload, 15s
    Memory Management  :memory, after marker, 3s

    section Text Processing
    Chunking Strategy  :chunk, after memory, 5s
    Embedding Gen     :embed, after chunk, 8s

    section Vector Storage
    FAISS Storage     :faiss, after embed, 2s
    ChromaDB Storage  :chroma, after embed, 4s
    Index Update      :index, after chroma, 3s

    section Query Processing
    Query Analysis    :milestone, query, after index, 0
    Vector Search     :search, after query, 1s
    Context Retrieval :retrieve, after search, 2s
    Model Inference   :inference, after retrieve, 5s
    Response Gen      :response, after inference, 3s

    section Total Timeline
    Complete Process  :milestone, complete, after response, 0
```

### 🔧 Pipeline Configuration

#### Environment-Specific Pipeline Settings

```yaml
# pipeline-config.yml
pipeline:
  stages:
    pdf_processing:
      timeout: 300s
      memory_limit: 4GB
      cache_enabled: true
      fallback_enabled: true

    text_processing:
      chunking_strategy: "hybrid"
      chunk_size: 512
      chunk_overlap: 50
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

    vector_storage:
      primary_backend: "chromadb"
      fallback_backend: "faiss"
      hybrid_threshold: 10000 # vectors
      collection_ttl: 7d

    model_inference:
      primary_provider: "groq"
      fallback_provider: "ollama"
      timeout: 30s
      max_retries: 3

  monitoring:
    metrics_enabled: true
    health_checks: true
    performance_tracking: true
    error_logging: true
```

### 🚨 Pipeline Error Handling

#### Error Recovery Strategies

```mermaid
flowchart TD
    START([Pipeline Start]) --> PROCESS{Processing Stage}

    PROCESS --> |PDF Processing Error| PDF_ERROR[PDF Processing Failed]
    PROCESS --> |Embedding Error| EMB_ERROR[Embedding Failed]
    PROCESS --> |Storage Error| STORE_ERROR[Storage Failed]
    PROCESS --> |Inference Error| INF_ERROR[Model Inference Failed]

    PDF_ERROR --> PDF_FALLBACK{Fallback Available?}
    PDF_FALLBACK --> |Yes| PYPDF[PyPDF2 Fallback]
    PDF_FALLBACK --> |No| FAIL[Pipeline Failed]

    EMB_ERROR --> EMB_RETRY{Retry Count < 3?}
    EMB_RETRY --> |Yes| RETRY_EMB[Retry Embedding]
    EMB_RETRY --> |No| FAIL

    STORE_ERROR --> STORE_FALLBACK{Alternative Storage?}
    STORE_FALLBACK --> |ChromaDB → FAISS| FAISS_STORE[Store in FAISS]
    STORE_FALLBACK --> |FAISS → ChromaDB| CHROMA_STORE[Store in ChromaDB]
    STORE_FALLBACK --> |No Alternative| FAIL

    INF_ERROR --> INF_FALLBACK{Provider Fallback?}
    INF_FALLBACK --> |Groq → Ollama| OLLAMA[Use Ollama]
    INF_FALLBACK --> |Ollama → Groq| GROQ[Use Groq]
    INF_FALLBACK --> |No Provider| FAIL

    PYPDF --> SUCCESS[Pipeline Success]
    RETRY_EMB --> PROCESS
    FAISS_STORE --> SUCCESS
    CHROMA_STORE --> SUCCESS
    OLLAMA --> SUCCESS
    GROQ --> SUCCESS

    style SUCCESS fill:#c8e6c9
    style FAIL fill:#ffcdd2
```

Bu comprehensive pipeline şeması, mikroservis mimarisindeki tüm veri işleme akışlarını, state transitions'ları, error handling stratejilerini ve performance monitoring'i detaylı olarak göstermektedir.

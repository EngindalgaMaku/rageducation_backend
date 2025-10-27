# PDF Marker and Sentence Transformers Dependency Restoration Report

## Problem Identified

The user reported that PDF marker was removed but is still needed, along with sentence transformers. Analysis revealed:

### Root Cause

- `marker-pdf>=0.2.15` and `sentence-transformers>=2.2.2` were present in `requirements.txt`
- But were **missing** from `requirements-minimal.txt` and `requirements-clean.txt`
- Docker containers use these minimal requirement files, causing import failures

### Impact Analysis

1. **PDF Processing Broken**: Enhanced PDF processor couldn't import marker dependencies
2. **Embeddings Fallback Broken**: Cloud LLM providers rely on sentence transformers for embeddings
3. **Re-ranking Broken**: CrossEncoder from sentence transformers used in RAG re-ranking

## Dependencies Restored

### Updated Files

#### `requirements-minimal.txt`

Added essential dependencies:

```
# PDF processing - Essential for document processing
PyPDF2
marker-pdf>=0.2.15

# Embeddings - Required for cloud LLM fallback embeddings
sentence-transformers>=2.2.2

# System monitoring - Required by enhanced PDF processor
psutil>=5.9.0

# Vector store for embeddings
faiss-cpu>=1.7.4
```

#### `requirements-clean.txt`

Added the same essential dependencies to clean API requirements

#### `Dockerfile.api`

Enhanced system dependencies for marker-pdf:

```dockerfile
# System deps for marker-pdf and other dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       git \
       libgl1-mesa-glx \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender-dev \
       libgomp1 \
       libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*
```

## Affected Components Verified

### PDF Processing

- `src/document_processing/enhanced_pdf_processor.py` - Primary marker usage
- `src/document_processing/pdf_processor.py` - Fallback processor

### Embeddings

- `src/embedding/embedding_generator.py` - Sentence transformers integration
- `src/rag/re_ranker.py` - CrossEncoder usage

### API Integration

- `src/app_logic.py` - Cloud LLM embedding fallbacks
- `src/api/main.py` - Multiple embedding generation calls

## Testing

Created `test_pdf_marker_fix.py` to verify:

1. Marker PDF imports
2. Sentence transformers imports
3. Enhanced PDF processor initialization
4. Embedding generation functionality
5. Actual PDF file processing

## Next Steps

1. Rebuild Docker containers with updated requirements
2. Test PDF processing with actual files
3. Verify cloud LLM + local embeddings workflow
4. Monitor performance impact of restored dependencies

## Environment Variables

For optimal marker-pdf performance, consider setting:

```bash
MARKER_TIMEOUT_SECONDS=1800  # 30 minutes for large PDFs
MARKER_MAX_MEMORY_MB=8192    # 8GB memory limit
MARKER_USE_GPU=true          # Enable GPU if available
```

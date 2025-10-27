# Cloud Run Deployment Configuration for RAG3 API

## Resource Configuration Options

### 1. CPU & Memory Specifications

```
CPU Options:
- 0.08 to 8 vCPUs (recommended: 2-4 vCPUs for AI workloads)
- CPU allocation: "Always allocated" (for consistent performance)

Memory Options:
- 128MB to 32GB (recommended: 8-16GB for PDF+AI processing)
- Memory allocation: High for marker-pdf and sentence-transformers
```

### 2. Request & Timeout Settings

```
Request Timeout: 3600 seconds (1 hour) - ideal for large PDF processing
Max Concurrent Requests: 80-100 per container instance
Min/Max Instances: 0 to 100 (serverless scaling)
```

### 3. Environment Variables

```
GOOGLE_CLOUD_PROJECT=your-project-id
MARKER_API_KEY=your-marker-key
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
```

### 4. Service Configuration

```
Service Name: rag3-api
Region: europe-west1 (Amsterdam - closest to Turkey)
Traffic: 100% to latest revision
Port: 8000 (FastAPI default)
```

## RAG3 Specific Recommendations

### Memory & CPU for Your Use Case:

```
PDF Processing (marker-pdf): 4-8GB RAM minimum
AI Models (sentence-transformers): 2-4GB RAM
Vector Operations (FAISS): 2-4GB RAM
CUDA Libraries: 2GB+ RAM

Total Recommended: 12-16GB RAM, 4 vCPUs
```

### Container Settings:

```
Container Image: gcr.io/[PROJECT-ID]/rag3-api:latest
Container Port: 8000
Health Check: /health endpoint
Startup Probe: /health with 300s timeout
```

### Cost Estimation (12GB RAM, 4 vCPUs):

```
Light Usage (100 req/day): ~$25/month
Medium Usage (1000 req/day): ~$120/month
Heavy Usage (5000 req/day): ~$450/month
```

## Deployment Commands

### 1. Build & Push Docker Image:

```bash
# Tag for Google Container Registry
docker tag rag3-api-fixed gcr.io/[PROJECT-ID]/rag3-api:latest

# Push to GCR
docker push gcr.io/[PROJECT-ID]/rag3-api:latest
```

### 2. Deploy to Cloud Run:

```bash
gcloud run deploy rag3-api \
  --image gcr.io/[PROJECT-ID]/rag3-api:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 12Gi \
  --cpu 4 \
  --timeout 3600 \
  --max-instances 10 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=[PROJECT-ID]" \
  --port 8000
```

### 3. Custom Domain (Optional):

```bash
gcloud run domain-mappings create \
  --service rag3-api \
  --domain api.your-domain.com \
  --region europe-west1
```

## Advanced Configuration Access

### Through Console:

1. Select "Public API" template
2. Click "Advanced Settings"
3. Configure CPU/Memory under "Resources"
4. Set Environment Variables under "Variables"
5. Configure Timeouts under "Connections"

### Through CLI:

```bash
gcloud run services replace service.yaml
```

## Monitoring & Logging

```
Cloud Logging: Automatic request/error logs
Cloud Monitoring: CPU, Memory, Request metrics
Error Reporting: Automatic error tracking
Cloud Trace: Request tracing for performance
```

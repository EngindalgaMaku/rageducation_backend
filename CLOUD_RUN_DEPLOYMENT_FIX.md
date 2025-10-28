# Cloud Run Deployment Fix Guide

## Problem Analysis

The Cloud Run deployment was failing with the error:

```
The user-provided container failed to start and listen on the port defined provided by the PORT=8080 environment variable within the allocated timeout.
```

## Root Cause

The main issue was in `startup.py` where `os.system()` was being used to start the API servers. This approach doesn't properly handle Cloud Run's port configuration and process lifecycle requirements.

## Solution Implemented

### 1. Fixed startup.py

**Before (Problematic):**

```python
def start_minimal_api():
    os.system("python src/api/main_minimal.py")
```

**After (Fixed):**

```python
def start_minimal_api():
    import uvicorn
    from src.api.main_minimal import app

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
```

**Key Changes:**

- Direct import and execution instead of `os.system()`
- Proper handling of Cloud Run's `PORT` environment variable
- Correct host binding (`0.0.0.0` required for Cloud Run)

### 2. Added Missing Package File

Created `src/api/__init__.py` to make the API directory a proper Python package.

### 3. Enhanced Cloud Build Configuration

Updated `cloudbuild.yaml` with:

- `FORCE_MINIMAL_API=true` environment variable to ensure reliable startup
- `--no-cpu-throttling` for better performance during startup
- `--execution-environment=gen2` for improved reliability
- Proper `PYTHONPATH=/app` setting
- `GROQ_API_KEY` for fast LLM inference and chunking operations

## Deployment Instructions

### Option 1: Deploy via Google Cloud Build

```bash
gcloud builds submit --config cloudbuild.yaml
```

### Option 2: Deploy via gcloud CLI

```bash
# Build the container
docker build -f Dockerfile.api -t gcr.io/YOUR_PROJECT_ID/rageducation-backend .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/rageducation-backend

# Deploy to Cloud Run
gcloud run deploy rageducation-backend \
  --image gcr.io/YOUR_PROJECT_ID/rageducation-backend \
  --region europe-west1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --timeout 3600 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 5 \
  --no-cpu-throttling \
  --execution-environment gen2 \
  --set-env-vars "ENVIRONMENT=production,FORCE_MINIMAL_API=true,PYTHONPATH=/app,GROQ_API_KEY=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" \
  --port 8080
```

## Verification Steps

After deployment, test the service:

1. **Health Check:**

   ```bash
   curl https://YOUR_SERVICE_URL/health
   ```

2. **Root Endpoint:**

   ```bash
   curl https://YOUR_SERVICE_URL/
   ```

3. **Test Endpoint:**
   ```bash
   curl https://YOUR_SERVICE_URL/test
   ```

Expected response for all endpoints should be JSON with status information.

## Troubleshooting

### If Deployment Still Fails:

1. **Check Container Logs:**

   ```bash
   gcloud logs read --service=rageducation-backend --limit=50
   ```

2. **Verify Environment Variables:**

   ```bash
   gcloud run services describe rageducation-backend --region=europe-west1
   ```

3. **Test Local Container:**
   ```bash
   docker build -f Dockerfile.api -t test-backend .
   docker run -p 8080:8080 -e PORT=8080 test-backend
   ```

### Common Issues and Solutions:

- **Import Errors:** Ensure all `__init__.py` files are present
- **Port Binding:** Always use `host="0.0.0.0"` for Cloud Run
- **Memory Issues:** The container is configured with 16GB RAM for heavy model loading
- **Startup Timeout:** Use `FORCE_MINIMAL_API=true` for faster startup

## Performance Optimization

The current configuration prioritizes startup reliability:

- Forces minimal API mode to avoid complex dependency loading
- Uses 16GB memory for model caching
- Disables CPU throttling for faster startup
- Uses Generation 2 execution environment

For production use, you can gradually enable more features by removing `FORCE_MINIMAL_API=true` once the basic deployment is stable.

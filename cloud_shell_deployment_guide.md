# Google Cloud Shell Deployment Guide

## RAG3 API with PDF Marker & Sentence Transformers

This guide helps you deploy your RAG API to Google Cloud Run using Cloud Shell with all dependencies including PDF marker and sentence transformers.

## Prerequisites ✅

- Google Cloud account with billing enabled
- Google Cloud project created
- Cloud Shell access (available in Google Cloud Console)

## Step 1: Upload Project Files to Cloud Shell

### Option A: Upload via Cloud Shell Editor

1. Open **Google Cloud Shell** in your browser
2. Click the **Editor** button (pencil icon) in Cloud Shell toolbar
3. Create a new folder: `mkdir rag3-for-colab && cd rag3-for-colab`
4. Upload the following essential files using the **File > Upload Files** option:
   - `src/` (entire folder with all Python files)
   - `data/` (entire folder with uploads and vector database)
   - `requirements-minimal.txt` ✨ (contains restored PDF marker & sentence transformers)
   - `Dockerfile.api` ✨ (updated with system dependencies)
   - `deploy_to_cloud_shell.sh` (automated deployment script)
   - `.env` files (if you have environment configurations)

### Option B: Upload via Cloud Shell Terminal

1. Open **Google Cloud Shell Terminal**
2. Create project directory: `mkdir rag3-for-colab && cd rag3-for-colab`
3. Use `gcloud` to upload files from your local machine (requires Google Cloud SDK)

### Key Files for Cloud Build Success:

- **requirements-minimal.txt**: ✨ Now includes `marker-pdf>=0.2.15`, `sentence-transformers>=2.2.2`
- **Dockerfile.api**: ✨ Enhanced with system packages for PDF processing
- **src/**: All your RAG application Python source code
- **data/**: Document uploads and vector database files

## Step 2: Deploy to Google Cloud Run

### Automated Deployment (Recommended)

1. Make the deployment script executable:

   ```bash
   chmod +x deploy_to_cloud_shell.sh
   ```

2. Run the automated deployment:
   ```bash
   ./deploy_to_cloud_shell.sh
   ```

### Manual Deployment Steps

If you prefer manual control, run these commands one by one:

```bash
# 1. Set your project (replace with your actual project ID)
gcloud config set project YOUR_PROJECT_ID

# 2. Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 3. Build Docker image with Cloud Build
PROJECT_ID=$(gcloud config get-value project)
gcloud builds submit --tag gcr.io/${PROJECT_ID}/rag3-api --file Dockerfile.api .

# 4. Deploy to Cloud Run (Academic Optimized Settings)
gcloud run deploy rag3-api \
  --image gcr.io/${PROJECT_ID}/rag3-api \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 1800 \
  --concurrency 2
```

## Step 3: Test Your Deployment

After successful deployment, test these endpoints:

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe rag3-api --region=europe-west1 --format='value(status.url)')
echo "Service URL: $SERVICE_URL"

# Test health check
curl "${SERVICE_URL}/health"

# Test available models
curl "${SERVICE_URL}/models"

# Test PDF processing (if you have documents uploaded)
curl -X POST "${SERVICE_URL}/configure_and_process" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemini-pro", "session_id": "test-session"}'
```

## Academic Research Configuration 🎓

Your deployment is optimized for academic research with:

- **Scale to Zero**: Costs nothing when not in use
- **4GB Memory**: Handles PDF processing and embeddings
- **2 vCPUs**: Adequate for moderate traffic
- **Max 3 instances**: Cost control for research budgets
- **30-minute timeout**: Allows long PDF processing tasks
- **Europe West 1**: Closest region to Turkey

### Expected Monthly Costs:

- **Light usage** (testing, demos): $4-8
- **Regular development**: $15-25
- **Heavy research periods**: $40-60

Compare to always-on server: **$75+/month**

## Troubleshooting 🔧

### Build Issues:

```bash
# Check build logs
gcloud builds list --limit=5

# View specific build log
gcloud builds log BUILD_ID
```

### Service Issues:

```bash
# Check service status
gcloud run services describe rag3-api --region=europe-west1

# View logs
gcloud logging read 'resource.type=cloud_run_revision' --limit=50
```

### PDF Marker Issues:

- ✅ System packages installed: `libgl1-mesa-dev`, `libglib2.0-0`, `ffmpeg`
- ✅ Python package restored: `marker-pdf>=0.2.15`
- ✅ Memory allocated: 4GB (sufficient for PDF processing)

### Sentence Transformers Issues:

- ✅ Package restored: `sentence-transformers>=2.2.2`
- ✅ Fallback embedding model available
- ✅ PyTorch dependencies included

## Managing Your Service 📊

### Update Service:

```bash
# Rebuild and redeploy after code changes
gcloud builds submit --tag gcr.io/${PROJECT_ID}/rag3-api --file Dockerfile.api .
gcloud run services update rag3-api --image gcr.io/${PROJECT_ID}/rag3-api --region=europe-west1
```

### Scale Service:

```bash
# Increase for higher traffic
gcloud run services update rag3-api \
  --max-instances 5 \
  --region=europe-west1

# Reduce for cost savings
gcloud run services update rag3-api \
  --max-instances 1 \
  --region=europe-west1
```

### Monitor Usage:

```bash
# View service metrics in Google Cloud Console
# Navigate to: Cloud Run > rag3-api > Metrics
```

## Success Indicators ✅

Your deployment is working correctly when:

1. ✅ Health endpoint returns `{"status": "healthy"}`
2. ✅ Models endpoint lists available LLM models
3. ✅ PDF processing completes without marker errors
4. ✅ Sentence transformers load for local embeddings
5. ✅ Service scales down to zero when idle
6. ✅ Cold start takes <30 seconds

Your RAG3 API with PDF marker and sentence transformers is now production-ready on Google Cloud Run! 🚀

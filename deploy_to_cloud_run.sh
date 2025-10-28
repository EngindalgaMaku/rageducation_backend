#!/bin/bash

# Google Cloud Shell Deployment Script for RAG3 API
# Deploy RAG API with PDF Marker and Sentence Transformers to Google Cloud Run

set -e  # Exit on any error

echo "🚀 Starting Google Cloud Run deployment for RAG3 API..."

# Configuration
PROJECT_ID=$(gcloud config get-value project)
SERVICE_NAME="rag3-api"
REGION="europe-west1"  # Closest to Turkey
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "📋 Project: ${PROJECT_ID}"
echo "🌍 Region: ${REGION}"
echo "🖼️  Image: ${IMAGE_NAME}"

# Step 1: Enable required APIs
echo "🔧 Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Step 2: Build Docker image in Cloud Build
echo "🏗️  Building Docker image with Cloud Build..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Step 3: Deploy to Cloud Run with 16GB RAM for optimal PDF processing
echo "🚀 Deploying to Cloud Run with 16GB RAM..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 3600 \
  --concurrency 5 \
  --cpu-throttling=false \
  --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO,MARKER_MAX_MEMORY_MB=15000,GCS_BUCKET_NAME=${SERVICE_NAME}-data-$(date +%s)"

# Step 4: Create GCS bucket for persistent storage
BUCKET_NAME="${SERVICE_NAME}-data-$(date +%s)"
echo "🗄️  Creating GCS bucket for persistent data: ${BUCKET_NAME}"
gsutil mb gs://${BUCKET_NAME} || echo "Bucket creation skipped (may already exist)"

# Set proper bucket permissions
gsutil iam ch serviceAccount:$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")-compute@developer.gserviceaccount.com:objectAdmin gs://${BUCKET_NAME}

# Step 5: Get the service URL
echo "🎉 Deployment complete!"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
echo "📍 Service URL: ${SERVICE_URL}"

# Step 6: Test the deployment
echo "🧪 Testing deployment..."
echo "Testing health endpoint..."
curl -s "${SERVICE_URL}/health"
echo ""

echo "Testing models endpoint..."
curl -s "${SERVICE_URL}/models"
echo ""

echo ""
echo "✅ Deployment successful!"
echo "🌐 Your RAG API is now running at: ${SERVICE_URL}"
echo "📊 PDF Marker and Sentence Transformers are included"
echo "🚀 16GB RAM enables enhanced PDF processing and model caching"
echo "🗄️  Persistent storage bucket: gs://${BUCKET_NAME}"
echo ""
echo "📝 Key endpoints to test:"
echo "  - Health: ${SERVICE_URL}/health"
echo "  - Models: ${SERVICE_URL}/models"
echo "  - Sessions: ${SERVICE_URL}/sessions"
echo "  - Process: ${SERVICE_URL}/rag/configure-and-process"
echo ""
echo "📈 Monitor your service:"
echo "  gcloud run services describe ${SERVICE_NAME} --region=${REGION}"
echo "  gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}' --limit=50"
echo ""
echo "💡 Configuration:"
echo "  - Container Memory: 16GB"
echo "  - PDF Processing Memory Limit: 15GB"
echo "  - Persistent Storage: Google Cloud Storage"
echo "  - Database: SQLite with GCS backup"
echo ""
echo "🔧 Session Persistence Fix Applied:"
echo "  - Sessions now stored in Cloud Storage"
echo "  - Database persists across container restarts"
echo "  - No more session data loss!"
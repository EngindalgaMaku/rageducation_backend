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

# Step 3: Deploy to Cloud Run with academic-optimized settings
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 4 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 1800 \
  --concurrency 2 \
  --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO"

# Step 4: Get the service URL
echo "🎉 Deployment complete!"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
echo "📍 Service URL: ${SERVICE_URL}"

# Step 5: Test the deployment
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
echo ""
echo "📝 Key endpoints to test:"
echo "  - Health: ${SERVICE_URL}/health"
echo "  - Models: ${SERVICE_URL}/models"
echo "  - Process: ${SERVICE_URL}/configure_and_process"
echo ""
echo "📈 Monitor your service:"
echo "  gcloud run services describe ${SERVICE_NAME} --region=${REGION}"
echo "  gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}' --limit=50"
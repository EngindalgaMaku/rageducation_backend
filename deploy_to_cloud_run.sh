#!/bin/bash

# Cloud Run Deployment Script for RAG3 API
# Run this in Google Cloud Shell after uploading your files

set -e

echo "🚀 Starting RAG3 API Deployment to Cloud Run..."

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
echo "📝 Project ID: $PROJECT_ID"

# Set region
REGION="europe-west1"
SERVICE_NAME="rag3-api"
IMAGE_NAME="gcr.io/$PROJECT_ID/rag3-api-fixed"

echo "🏗️  Building container image..."
gcloud builds submit --tag $IMAGE_NAME .

echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 5 \
  --max-instances 3 \
  --min-instances 0 \
  --timeout 1800s \

echo "✅ Deployment completed!"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
echo "🌐 Service URL: $SERVICE_URL"

echo "🧪 Testing deployment..."
curl -s "$SERVICE_URL/health" || echo "Health check failed, but service might still be starting..."

echo ""
echo "🎉 Deployment Summary:"
echo "   - Project: $PROJECT_ID"
echo "   - Service: $SERVICE_NAME"
echo "   - Region: $REGION"
echo "   - URL: $SERVICE_URL"
echo "   - Memory: 4GB"
echo "   - CPU: 2 vCPUs"
echo "   - Max Instances: 3"
echo "   - Academic Cost Optimized: ✅"
echo ""
echo "🔗 Test your API:"
echo "   Health: $SERVICE_URL/health"
echo "   Models: $SERVICE_URL/models"
echo "   Configure: $SERVICE_URL/configure_and_process"
echo ""
echo "💰 Expected monthly cost: $4-8 for typical thesis usage"
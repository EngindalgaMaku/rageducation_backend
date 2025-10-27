#!/bin/bash

# Cloud Run - Optimized Deployment for RAG3
# Cost: $8-15/month for moderate usage

PROJECT_ID="your-project-id"  # Bu satırı değiştirin
SERVICE_NAME="rag3-api"
REGION="europe-west1"

echo "🚀 Deploying RAG3 to Cloud Run (Optimized for Cost)..."

# Build and push to Google Container Registry
echo "📦 Building and pushing Docker image..."
docker tag rag3-api-fixed gcr.io/${PROJECT_ID}/rag3-api:latest
docker push gcr.io/${PROJECT_ID}/rag3-api:latest

# Deploy to Cloud Run with cost-optimized settings
echo "🌟 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/rag3-api:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 1800 \
  --max-instances 5 \
  --min-instances 0 \
  --concurrency 10 \
  --cpu-throttling=false \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
  --port 8000

echo "✅ Deployment complete!"
echo "📊 Expected monthly cost: $8-20 (depending on usage)"
echo "🔗 Your API will be available at:"
gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)"

# Enable logging and monitoring
echo "📈 Setting up monitoring..."
echo "Check usage: https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics"
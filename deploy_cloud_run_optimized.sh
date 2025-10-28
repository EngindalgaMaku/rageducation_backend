#!/bin/bash

# Cloud Run - Optimized Deployment for RAG3 with 16GB RAM
# Cost: $20-40/month for moderate usage (increased due to higher memory)

PROJECT_ID="your-project-id"  # Bu satırı değiştirin
SERVICE_NAME="rag3-api"
REGION="europe-west1"

echo "🚀 Deploying RAG3 to Cloud Run (16GB RAM for PDF Processing & Model Caching)..."

# Build and push to Google Container Registry
echo "📦 Building and pushing Docker image..."
docker tag rag3-api-fixed gcr.io/${PROJECT_ID}/rag3-api:latest
docker push gcr.io/${PROJECT_ID}/rag3-api:latest

# Deploy to Cloud Run with 16GB memory for optimal performance
echo "🌟 Deploying to Cloud Run with 16GB RAM..."
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/rag3-api:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --timeout 3600 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 5 \
  --cpu-throttling=false \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},MARKER_MAX_MEMORY_MB=15000" \
  --port 8000

echo "✅ Deployment complete!"
echo "📊 Expected monthly cost: $20-50 (depending on usage - increased due to 16GB memory)"
echo "🔗 Your API will be available at:"
gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)"

# Enable logging and monitoring
echo "📈 Setting up monitoring..."
echo "Check usage: https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics"
echo "💡 16GB RAM enables better PDF processing and TensorFlow model caching"
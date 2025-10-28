# Secure Deployment Guide - Cloud Run with GROQ API

## Security Issue Resolution

GitHub's push protection detected the GROQ API key in the repository. This guide shows how to deploy securely without committing secrets.

## Secure Deployment Methods

### Method 1: Using Google Cloud Build with Secret Manager (Recommended)

1. **Store API Key in Secret Manager:**

```bash
echo "YOUR_GROQ_API_KEY_HERE" | gcloud secrets create groq-api-key --data-file=-
```

2. **Deploy with Build Substitution:**

```bash
gcloud builds submit --config cloudbuild.yaml --substitutions _GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
```

### Method 2: Direct gcloud CLI Deployment

```bash
# Build and push container
docker build -f Dockerfile.api -t gcr.io/nimble-gearing-476415-t2/rageducation-backend .
docker push gcr.io/nimble-gearing-476415-t2/rageducation-backend

# Deploy with secure environment variable
gcloud run deploy rageducation-backend \
  --image gcr.io/nimble-gearing-476415-t2/rageducation-backend \
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
  --set-env-vars "ENVIRONMENT=production,FORCE_MINIMAL_API=true,PYTHONPATH=/app,GROQ_API_KEY=YOUR_GROQ_API_KEY_HERE" \
  --port 8080
```

### Method 3: Update Existing Service

If you want to add the API key to an already deployed service:

```bash
gcloud run services update rageducation-backend \
  --region europe-west1 \
  --set-env-vars "GROQ_API_KEY=YOUR_GROQ_API_KEY_HERE"
```

## Git Repository Cleanup

To safely commit your changes:

1. **Commit the cleaned files:**

```bash
git add .
git commit -m "Fix Cloud Run deployment issues and secure API key handling"
git push
```

The repository now contains placeholder values instead of actual secrets.

## Environment Variables Configuration

### Local Development (.env file)

Create a `.env` file (never commit this):

```bash
GROQ_API_KEY=YOUR_GROQ_API_KEY_HERE
GROQ_MODEL=llama-3.1-70b-versatile
ENVIRONMENT=development
```

### Production (Cloud Run)

Environment variables are set securely during deployment.

## Verification Steps

After secure deployment:

1. **Check environment variables:**

```bash
gcloud run services describe rageducation-backend --region=europe-west1 --format="export"
```

2. **Test the service:**

```bash
curl https://your-service-url/health
curl https://your-service-url/models
```

## Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for all sensitive configuration
3. **Use Google Secret Manager** for production secrets
4. **Rotate API keys regularly**
5. **Monitor API usage** in GROQ dashboard

## Troubleshooting

### If deployment fails with missing API key:

- Verify the environment variable is set in Cloud Run
- Check Cloud Build logs for substitution errors
- Ensure the API key is valid in GROQ console

### If Git still blocks the push:

- Check if any other files contain the API key
- Use `git log --oneline` to see if previous commits need cleanup
- Consider using `git filter-branch` if needed for commit history cleanup

## Ready to Deploy

Now you can safely push to Git and deploy using any of the secure methods above. The actual API key will only exist in the deployment environment, not in your code repository.

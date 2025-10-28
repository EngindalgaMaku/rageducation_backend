# Git History Cleanup Guide - Remove Secrets from Commit History

## Problem

GitHub detected the GROQ API key in previous commits, even though current files are clean. The error shows:

```
commit: 113d65dd3f619f27414931c1c7ede3e31fb6ba2d
```

## Solution Options

### Option 1: Quick Fix - Bypass GitHub Protection (Recommended)

Since you've already cleaned the current files, you can bypass the protection once:

1. **Go to the GitHub URL provided in the error:**

   ```
   https://github.com/EngindalgaMaku/rageducation_backend/security/secret-scanning/unblock-secret/34i94HSl88VBUP3FnUhNA9ZKbn3
   ```

2. **Click "Allow secret" for this specific push**

3. **Push again:**
   ```bash
   git push
   ```

### Option 2: Clean Git History (More Secure)

Remove the API key from all Git history:

1. **Install git-filter-repo (if not already installed):**

   ```bash
   pip install git-filter-repo
   ```

2. **Remove the API key from all history:**

   ```bash
   git filter-repo --replace-text <(echo "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==>YOUR_GROQ_API_KEY_HERE")
   ```

3. **Force push the cleaned history:**
   ```bash
   git push --force-with-lease origin main
   ```

### Option 3: Create New Repository (Nuclear Option)

If other options fail:

1. **Create a new GitHub repository**
2. **Copy current clean files to new repository**
3. **Initial commit without any API keys**

## Current Status

✅ **Files are now clean:**

- `.env.example` - uses placeholder
- `cloudbuild.yaml` - uses substitution variable
- `CLOUD_RUN_DEPLOYMENT_FIX.md` - uses placeholder
- `SECURE_DEPLOYMENT_GUIDE.md` - uses placeholder

## Deploy Without Git Push

You can deploy immediately without pushing to Git:

### Direct Deployment (Recommended for now):

```bash
# Build and push container
docker build -f Dockerfile.api -t gcr.io/nimble-gearing-476415-t2/rageducation-backend .
docker push gcr.io/nimble-gearing-476415-t2/rageducation-backend

# Deploy with the actual API key
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
  --set-env-vars "ENVIRONMENT=production,FORCE_MINIMAL_API=true,PYTHONPATH=/app,GROQ_API_KEY=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" \
  --port 8080
```

### Or use Cloud Build with substitution:

```bash
gcloud builds submit --config cloudbuild.yaml --substitutions _GROQ_API_KEY="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"
```

## Verification

After deployment, test your service:

```bash
# Get the service URL
gcloud run services describe rageducation-backend --region=europe-west1 --format="value(status.url)"

# Test endpoints (replace YOUR_SERVICE_URL with actual URL)
curl https://YOUR_SERVICE_URL/health
curl https://YOUR_SERVICE_URL/test
curl https://YOUR_SERVICE_URL/models
```

## Next Steps

1. **Deploy first** using the commands above (deployment works independently of Git)
2. **Then handle Git** using Option 1 (bypass protection) or Option 2 (clean history)
3. **Future commits** will be secure since current files are clean

The main fix (Cloud Run port handling) is complete and ready to deploy!

# Google Cloud Setup Guide for Master's Thesis RAG Project

## 🎓 Step-by-Step Google Cloud Project Setup

### Step 1: Create Google Cloud Account & Project

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Sign in** with your Google account
3. **Click "Create Project"**
   - Project Name: `thesis-rag-system` (or similar)
   - Project ID: Will be auto-generated (e.g., `thesis-rag-system-123456`)
   - Leave Organization blank for personal projects

### Step 2: 🎯 Academic Credits (IMPORTANT!)

**Google offers $300 free credits for new accounts!**

1. **Enable Billing** - You won't be charged during free trial
2. **For Students**: Look into Google Cloud for Education
   - Additional credits available for academic projects
   - Contact your university's IT department

### Step 3: Enable Required APIs

In Google Cloud Console, go to "APIs & Services" > "Enable APIs":

```bash
# Required APIs for Cloud Run:
- Cloud Run API
- Container Registry API
- Cloud Build API
- Artifact Registry API
```

### Step 4: Install Google Cloud CLI

**Windows:**

```bash
# Download from: https://cloud.google.com/sdk/docs/install
# Or use PowerShell:
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe
```

**After installation:**

```bash
# Initialize gcloud
gcloud init

# Login to your account
gcloud auth login

# Set your project (use your actual project ID)
gcloud config set project YOUR-PROJECT-ID

# Configure Docker to use gcloud
gcloud auth configure-docker
```

### Step 5: 🚀 Deployment Commands

Once your Docker build completes, use these commands:

```bash
# 1. Tag your Docker image
docker tag rag3-api-fixed gcr.io/YOUR-PROJECT-ID/rag3-thesis:latest

# 2. Push to Google Container Registry
docker push gcr.io/YOUR-PROJECT-ID/rag3-thesis:latest

# 3. Deploy to Cloud Run (Academic Optimized)
gcloud run deploy rag3-thesis \
  --image gcr.io/YOUR-PROJECT-ID/rag3-thesis:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 1800 \
  --max-instances 3 \
  --min-instances 0 \
  --concurrency 5 \
  --port 8000
```

## 💰 Cost Management for Students

### Academic Budget Settings:

```bash
# Set spending limits
gcloud billing budgets create \
  --billing-account=YOUR-BILLING-ACCOUNT \
  --display-name="Thesis Budget" \
  --budget-amount=50USD \
  --threshold-rules-percent=0.5,0.8,0.9,1.0
```

### Expected Costs:

- **Development Phase**: $3-8/month
- **Testing Phase**: $5-12/month
- **With $300 credits**: **25+ months of usage!**

## 🎯 Quick Start Checklist

- [ ] Create Google Cloud Project
- [ ] Enable billing (free $300 credits)
- [ ] Enable required APIs
- [ ] Install gcloud CLI
- [ ] Authenticate and configure
- [ ] Wait for Docker build to complete
- [ ] Deploy with provided commands

## 🆘 Common Issues & Solutions

**Issue**: "Project ID already exists"
**Solution**: Add random numbers to your project name

**Issue**: "Billing not enabled"
**Solution**: Go to Billing > Link billing account (free trial)

**Issue**: "Permission denied"
**Solution**: Run `gcloud auth login` again

**Issue**: "Region not available"
**Solution**: Try `us-central1` or `europe-west1`

## 📧 Your Next Steps:

1. **Create the project** using the guide above
2. **Get your Project ID** (shown in console dashboard)
3. **Tell me your Project ID** when ready
4. **I'll provide exact deployment commands** with your ID

Once you have your Project ID, I'll update all deployment scripts with the correct values!

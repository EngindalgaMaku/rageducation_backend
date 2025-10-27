# Cloud Run Configuration for Master's Thesis Research Project

## 🎓 Academic Research Usage Profile

### Project Context:

- **Master's Thesis Development**
- **Active Development Phase**
- **Research & Testing Workload**
- **Single Developer/Researcher Usage**

## 📊 Realistic Academic Usage Estimates

### Development Phase (Current):

- **Requests/month**: 2,000-5,000
- **Testing sessions**: Daily PDF processing
- **Research iterations**: Multiple document uploads
- **Expected cost**: **$3-8/month**

### Traffic Pattern:

- ✅ **"Single spike"** (best for development)
- **1-hour bursts** during coding/testing sessions
- **Scale to zero** between sessions
- Perfect for academic workflow

## ⚙️ Optimal Cloud Run Settings

```bash
# Academic Research Configuration
gcloud run deploy rag3-thesis \
  --memory 4Gi \
  --cpu 2 \
  --timeout 1800 \
  --max-instances 3 \
  --min-instances 0 \
  --concurrency 5 \
  --requests-per-month 3000
```

## 💰 Expected Costs (Academic Use):

### Light Development:

- **50 PDFs/week**: $2-4/month
- **Testing & debugging**: $1-2/month
- **Total**: **$3-6/month**

### Heavy Development:

- **200 PDFs/week**: $5-8/month
- **Continuous testing**: $2-3/month
- **Total**: **$7-11/month**

## 🎯 Calculator Settings for Thesis:

- **Requests**: 3,000/month
- **Traffic**: "Single spike"
- **Duration**: 1 hour bursts
- **Scale-to-zero**: Enabled

This will show approximately **$4-8/month** - perfect for academic budget!

## 📚 Benefits for Academic Research:

1. **Pay only when testing**
2. **Scale during intensive research**
3. **Zero cost during writing phases**
4. **Perfect for thesis timeline**

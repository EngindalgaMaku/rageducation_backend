# Cloud Run Realistic Usage Estimates for RAG3

## 📊 Recommended Settings

### Number of requests per month:

- **Small Institution**: 5,000-10,000 requests
- **Medium Institution**: 15,000-30,000 requests
- **Large Institution**: 50,000-100,000 requests

### Traffic Shape:

- ✅ **Daily peak and trough** (recommended)
  - Active during school hours (12 hours)
  - Idle during night/weekend (12 hours)
  - Perfect for educational use case

### Expected Monthly Costs:

- **Small (5K requests)**: $3-8/month
- **Medium (20K requests)**: $8-15/month
- **Large (80K requests)**: $15-30/month

## 💡 Cost Optimization Tips:

1. **Start small** - 5,000 requests first month
2. **Monitor usage** - adjust based on real data
3. **Use scale-to-zero** - saves 50-70% cost
4. **Peak hours only** - educational institutions benefit most

## 🎯 Recommended Calculator Settings:

- Requests: **10,000/month** (realistic start)
- Traffic: **Daily peak and trough**
- CPU: **2 vCPUs**
- Memory: **4GB**
- Min instances: **0** (scale-to-zero)
- Max instances: **5**

This will show approximately **$5-12/month** - much more realistic than $75!

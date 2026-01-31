# Deployment Guide

> **Version 0.3.0** | Step-by-step deployment instructions

## Quick Reference

| Environment | Command |
|-------------|---------|
| Local (mock) | `uvicorn src.main:app --port 8000` |
| Local (vLLM) | Start vLLM on 8080, then API on 8000 |
| Docker | `docker-compose up` |
| RunPod | See [RunPod Deployment](#runpod-deployment) |

---

## Local Development

### Mock Mode (No GPU)

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Test
curl http://localhost:8000/health
```

The server uses `MockVLLMClient` when vLLM is unavailable, returning echo responses.

### With vLLM Backend

```bash
# Terminal 1: Start vLLM
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8080

# Terminal 2: Start API
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Verify connection
curl http://localhost:8000/health
# Should show: "vllm_connected": true
```

---

## Docker Deployment

### With GPU

```bash
# Build and run
docker-compose up

# Or build separately
docker build -t inference-api -f Dockerfile .
docker build -t vllm-server -f Dockerfile.vllm .
```

### Development (No GPU)

```bash
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up
```

---

## RunPod Deployment

### Option 1: Manual Pod

1. **Create Pod**
   - Go to [runpod.io/console/pods](https://runpod.io/console/pods)
   - Select H100 80GB ($1.99-2.69/hr)
   - Use PyTorch template

2. **Setup Environment**
   ```bash
   # SSH into pod
   pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu124
   pip install vllm==0.11.0 lmcache==0.3.9

   # Clone repo
   git clone https://github.com/CyberBerto/inference-server-mvp.git
   cd inference-server-mvp
   pip install -r requirements.txt
   ```

3. **Start Services**
   ```bash
   # Terminal 1: vLLM
   export HUGGING_FACE_HUB_TOKEN=hf_xxxxx
   LMCACHE_CONFIG_FILE=configs/lmcache.yaml \
   vllm serve meta-llama/Llama-3.1-8B-Instruct \
       --port 8080 \
       --max-model-len 131072 \
       --gpu-memory-utilization 0.9 \
       --enable-chunked-prefill

   # Terminal 2: API
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```

4. **Expose Endpoint**
   - Note your pod's public IP
   - Or use RunPod's proxy URL

### Option 2: Serverless

1. **Create Template**
   - Go to [runpod.io/console/serverless](https://runpod.io/console/serverless)
   - Create new template with your Docker image

2. **Configure**
   - Set environment variables
   - Configure scaling rules
   - Set idle timeout

3. **Deploy**
   - Create endpoint from template
   - Get endpoint URL for OpenRouter

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env`:

```bash
# Required for production
MODEL_ID=your-org/your-model
MODEL_DISPLAY_NAME=Your Model Name
VLLM_BASE_URL=http://localhost:8080

# Pricing (USD per token)
PRICE_PER_PROMPT_TOKEN=0.000008
PRICE_PER_COMPLETION_TOKEN=0.000024

# Optional
MAX_CONTEXT_LENGTH=131072
QUANTIZATION=fp16
REQUEST_TIMEOUT=300
```

### LMCache Configuration

Edit `configs/lmcache.yaml`:

```yaml
local_cpu: true
max_local_cpu_size: 64  # GB

# Optional: disk offloading
local_disk: false
max_local_disk_size: 100
```

---

## OpenRouter Submission

### Prerequisites

- [ ] Server running with real vLLM backend
- [ ] Health endpoint returning `vllm_connected: true`
- [ ] Model discovery at `/api/v1/models`
- [ ] Successful chat completions

### Steps

1. Go to [openrouter.ai/how-to-list](https://openrouter.ai/how-to-list)
2. Submit your endpoint URL
3. Provide model metadata
4. Wait 1-4 weeks for approval

### Requirements

| Metric | Requirement |
|--------|-------------|
| Uptime | 95%+ for normal routing |
| Health endpoint | Must return valid JSON |
| API format | OpenAI-compatible |
| Streaming | SSE with proper format |

---

## Troubleshooting

### vLLM Won't Start

```bash
# Check CUDA
nvidia-smi
nvcc --version

# Check memory
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
```

### Connection Refused

```bash
# Check if vLLM is running
curl http://localhost:8080/health

# Check ports
netstat -tlnp | grep -E "8000|8080"
```

### Out of Memory

```bash
# Reduce context length
--max-model-len 65536

# Reduce batch size
--max-num-seqs 32

# Enable disk offloading in lmcache.yaml
local_disk: true
```

---

## Cost Estimates

| Configuration | Hourly | Monthly (24/7) |
|---------------|--------|----------------|
| H100 80GB (RunPod) | $1.99-2.69 | ~$1,430-1,940 |
| A100 80GB | $1.64 | ~$1,180 |
| Serverless | Variable | $50-500 |

---

## Next Steps

After deployment:

1. **v0.4.0**: Add authentication, rate limiting, metrics
2. **Enterprise**: See [docs/ENTERPRISE-INFRASTRUCTURE.md](docs/ENTERPRISE-INFRASTRUCTURE.md)
3. **OpenRouter**: Submit for listing

## Support

- [vLLM Docs](https://docs.vllm.ai)
- [LMCache Docs](https://docs.lmcache.ai)
- [RunPod Docs](https://docs.runpod.io)
- [OpenRouter Docs](https://openrouter.ai/docs)

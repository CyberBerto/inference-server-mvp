# Deployment Guide & v0.3 Testing Roadmap

> Current Version: **v0.2.0** (Code Review Complete)
> Next Version: **v0.3.0** (Real Deployment Testing)

---

## Current State (v0.2.0)

### What's Ready
- ✅ FastAPI server with OpenAI-compatible API
- ✅ vLLM client with streaming support
- ✅ Mock client for local testing
- ✅ Health monitoring & error handling
- ✅ OpenRouter-compatible model discovery
- ✅ Docker configurations (API + vLLM)
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Comprehensive test suite
- ✅ Full code documentation

### What's Placeholder
- 🔲 `.env` values (MODEL_ID, HF_TOKEN, pricing)
- 🔲 RunPod deployment automation
- 🔲 Real GPU testing
- 🔲 OpenRouter submission

---

## v0.3 Testing Roadmap

### Phase 1: Local Mock Testing (Day 1)
**Goal:** Verify API layer works correctly

```bash
# 1. Setup environment
cd inference-server-mvp
cp .env.example .env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 2. Run tests
pytest tests/ -v

# 3. Start server with mock backend
uvicorn src.main:app --reload --port 8000

# 4. Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/models
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}]}'
```

**Success Criteria:**
- [ ] All pytest tests pass
- [ ] Health endpoint returns valid JSON
- [ ] Models endpoint returns OpenRouter-compatible format
- [ ] Chat completions return mock responses
- [ ] Streaming endpoint returns SSE format

---

### Phase 2: Account Setup (Day 1-2)
**Goal:** Get access to required services

#### HuggingFace (10 min)
1. Create account at https://huggingface.co
2. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. Accept the Llama license agreement
4. Generate API token at https://huggingface.co/settings/tokens
5. Save token for `.env` file

#### RunPod (10 min)
1. Create account at https://runpod.io
2. Add payment method (credit card or crypto)
3. Note: H100 80GB = $1.99-2.69/hr
4. Generate API key at https://runpod.io/console/user/settings

#### GitHub (5 min)
1. Create new repository
2. Push this codebase
3. Enable GitHub Actions
4. Add secrets: `RUNPOD_API_KEY`, `HF_TOKEN`

---

### Phase 3: GPU Testing on RunPod (Day 2-3)
**Goal:** Validate real inference with vLLM + LMCache

#### Option A: Manual Pod Deployment
```bash
# 1. Deploy H100 80GB pod on RunPod
#    - Select PyTorch template
#    - Choose region with availability

# 2. SSH into pod and setup
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.11.0 lmcache==0.3.9

# 3. Start vLLM with LMCache
export HUGGING_FACE_HUB_TOKEN=hf_xxxxx
LMCACHE_CONFIG_FILE=configs/lmcache.yaml \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8080 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill

# 4. In second terminal, start API server
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000

# 5. Test from local machine (replace with pod IP)
curl http://<POD_IP>:8000/health
curl -X POST http://<POD_IP>:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Explain quantum computing in one sentence."}]}'
```

#### Option B: Docker Compose Deployment
```bash
# On RunPod pod with GPU
git clone <your-repo>
cd inference-server-mvp
cp .env.example .env
# Edit .env with real values

docker-compose up
```

**Success Criteria:**
- [ ] vLLM starts and loads model
- [ ] LMCache initializes (check logs for "LMCache enabled")
- [ ] Health endpoint shows `vllm_connected: true`
- [ ] Chat completion returns real inference
- [ ] Streaming works with token-by-token output
- [ ] 100K context request completes (test with long prompt)

---

### Phase 4: Performance Validation (Day 3-4)
**Goal:** Verify production readiness

#### Tests to Run
```bash
# 1. Latency test (short prompt)
time curl -X POST http://<POD_IP>:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'

# 2. Long context test (100K tokens)
# Generate a long prompt file first
curl -X POST http://<POD_IP>:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @long_prompt.json

# 3. Streaming latency (time to first token)
curl -X POST http://<POD_IP>:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Write a story"}],"stream":true}'

# 4. Concurrent requests
# Use tools like wrk, hey, or locust
```

**Success Criteria:**
- [ ] TTFT (Time to First Token) < 2s for short prompts
- [ ] 100K context processes without OOM
- [ ] Streaming keeps connection alive
- [ ] Error rate < 1% under load
- [ ] Health endpoint stays responsive

---

### Phase 5: Production Preparation (Day 4-5)
**Goal:** Ready for OpenRouter submission

#### Configuration Updates
```bash
# Update .env with production values
MODEL_ID=your-org/your-model-name
MODEL_DISPLAY_NAME=Your Model Display Name
ORGANIZATION_ID=your-org
PRICE_PER_PROMPT_TOKEN=0.000008    # $8/M tokens
PRICE_PER_COMPLETION_TOKEN=0.000024 # $24/M tokens
```

#### RunPod Serverless Setup
1. Create serverless template at https://runpod.io/console/serverless
2. Use Docker image from GitHub Container Registry
3. Configure environment variables
4. Set up auto-scaling rules
5. Get endpoint URL

#### OpenRouter Submission
1. Go to https://openrouter.ai/how-to-list
2. Submit endpoint URL
3. Provide model metadata
4. Wait 1-4 weeks for approval

---

## v0.3.0 Release Checklist

Before tagging v0.3.0:

- [ ] Phase 1: Local mock tests pass
- [ ] Phase 2: All accounts created
- [ ] Phase 3: GPU inference verified
- [ ] Phase 4: Performance validated
- [ ] Phase 5: Production config ready
- [ ] Update version in `pyproject.toml`, `main.py`, `__init__.py`
- [ ] Document any config changes
- [ ] Tag release: `git tag -a v0.3.0 -m "Real deployment testing complete"`

---

## Cost Estimates

| Phase | Resource | Cost |
|-------|----------|------|
| Phase 1-2 | Local only | $0 |
| Phase 3-4 | H100 80GB (~8 hrs) | ~$16-22 |
| Phase 5 | Serverless testing | ~$5-20 |
| **Total Testing** | | **~$25-50** |

| Production | Resource | Monthly |
|------------|----------|---------|
| Low traffic | Serverless | $50-200 |
| Medium traffic | 1x H100 pod | ~$1,430 |
| High traffic | Multi-GPU | $3,000+ |

---

## Troubleshooting

### vLLM won't start
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Check memory
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
```

### LMCache not working
```bash
# Verify config file exists
cat configs/lmcache.yaml

# Check environment variable
echo $LMCACHE_CONFIG_FILE

# Look for LMCache logs
# Should see: "LMCache connector initialized"
```

### Connection refused to vLLM
```bash
# Check if vLLM is running
curl http://localhost:8080/health

# Check the port
netstat -tlnp | grep 8080
```

### Out of memory
```bash
# Reduce context length
--max-model-len 65536  # 64K instead of 128K

# Reduce batch size
--max-num-seqs 32  # Down from 64

# Enable more aggressive offloading in lmcache.yaml
local_disk: true
max_local_disk_size: 100
```

---

## Support

- vLLM Docs: https://docs.vllm.ai
- LMCache Docs: https://docs.lmcache.ai
- RunPod Docs: https://docs.runpod.io
- OpenRouter: https://openrouter.ai/docs

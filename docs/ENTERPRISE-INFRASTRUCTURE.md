# Enterprise Infrastructure Research: Weka Token Warehouse & Neocloud

## Executive Summary

Your MVP (v0.3.0) uses a lightweight architecture optimized for rapid deployment:
- **vLLM** + **LMCache** (local CPU RAM) on **RunPod H100**

Scaling to enterprise requires addressing three domains:
1. **Storage**: Local CPU RAM → Distributed KV warehouse (Weka, VAST, Pure Storage)
2. **Infrastructure**: Spot instances → Dedicated clusters (DGX SuperPOD, CoreWeave)
3. **Operations**: Add compliance, multi-tenancy, SLA management

---

## Part 1: Weka Token Warehouse / Augmented Memory Grid

### What Is It?

Weka's **Augmented Memory Grid (AMG)** converts KV caches from transient GPU artifacts into durable, reusable assets in a persistent "token warehouse."

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Cluster (8x H100)                    │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│  │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│ │GPU 4│ │GPU 5│ │GPU 6│ │GPU 7│  │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘  │
│     │       │       │       │       │       │       │       │      │
│     └───────┴───────┴───────┼───────┴───────┴───────┴───────┘      │
│                             │                                       │
│                    GPUDirect Storage (RDMA)                         │
│                             │                                       │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Weka Parallel Filesystem                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Token Warehouse (KV Cache)                 │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
│  │  │ Prefill │  │ Prefill │  │ Prefill │  │ Prefill │  ...    │   │
│  │  │ Cache 1 │  │ Cache 2 │  │ Cache 3 │  │ Cache N │         │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Performance: 270 GB/s read throughput, microsecond latency          │
│  Capacity: 500GB - 10TB+ shared across cluster                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Capabilities

| Feature | Benefit |
|---------|---------|
| **GPUDirect Storage** | Zero-copy data transfer, RDMA fabric |
| **270 GB/s throughput** | Tested across 8x H100 |
| **Microsecond latency** | Near-DRAM speeds for cache hits |
| **Persistent cache** | Survives pod restarts, shared across replicas |
| **75x TTFT improvement** | Eliminates redundant prefill computation |

### Integration with LMCache

Weka is a **backend** for LMCache (not a replacement):

```yaml
# Current MVP: configs/lmcache.yaml
backend: local_cpu
max_local_cpu_size: 64  # GB, single-node limit

# Enterprise upgrade
backend: weka
weka_mds_endpoints:
  - "10.0.1.10:24753"
  - "10.0.1.11:24753"
max_cache_size_gb: 500  # Shared across cluster
enable_gpu_direct_storage: true
replication_factor: 3
```

---

## Part 2: Enterprise GPU Cloud Providers

### Comparison Matrix

| Provider | Tier | H100 Cost | SLA | Compliance | Best For |
|----------|------|-----------|-----|------------|----------|
| **RunPod** (current) | Spot | $1.99-2.69/hr | Best-effort | Limited | Dev, bursty workloads |
| **CoreWeave** | Enterprise | $2.50-4.00/hr | 99.9% | SOC2, HIPAA, FedRAMP | Mission-critical |
| **Lambda Labs** | Semi-managed | ~$2.50/hr | 95-99% | SOC2, GDPR | Training + inference |
| **DGX SuperPOD** | On-prem | Capex | Self-managed | Full control | 24/7 baseline, data sovereignty |

### Recommended Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Traffic (100%)                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼ (80%)               ▼ (15%)               ▼ (5%)
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   CoreWeave   │    │  Lambda Labs  │    │    RunPod     │
│   (Primary)   │    │   (Failover)  │    │    (Burst)    │
│               │    │               │    │               │
│ • 8x H100 ded │    │ • 2x H100 OD  │    │ • Spot H100   │
│ • 99.9% SLA   │    │ • 95% SLA     │    │ • Best-effort │
│ • SOC2/HIPAA  │    │ • SOC2        │    │ • Cheapest    │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Part 3: Enterprise Requirements Checklist

### Security & Compliance

| Requirement | Status in MVP | Enterprise Need |
|-------------|---------------|-----------------|
| **SOC2 Type II** | ❌ Not started | Required for enterprise sales |
| **HIPAA** | ❌ Not started | Required for healthcare |
| **GDPR** | ❌ Not started | Required for EU customers |
| **Data residency** | ❌ Not configurable | Region-specific deployment |
| **Encryption at rest** | ❌ Not implemented | AES-256 required |
| **Encryption in transit** | ⚠️ TLS available | mTLS for inter-service |
| **Audit logging** | ❌ TODO in code | Every request logged |

### SLA Tiers

| Uptime | Downtime/Year | Routing Priority |
|--------|---------------|------------------|
| 99.9% | 8.6 hours | Normal |
| 99.99% | 52 minutes | Premium |
| 99.999% | 5 minutes | Mission-critical |

### Multi-Tenancy Levels

| Level | Isolation | Cost Impact | Use Case |
|-------|-----------|-------------|----------|
| **Soft** | Shared GPU, rate-limited | +10% | Non-sensitive |
| **Hard** | Dedicated GPU, VLAN | +50% | Healthcare, financial |
| **Complete** | Dedicated pod, separate network | +100% | Government, defense |

---

## Part 4: Alternative Enterprise Storage

### Comparison

| Feature | Weka | VAST Data | Pure FlashBlade | NetApp ONTAP |
|---------|------|-----------|-----------------|--------------|
| **KV cache focus** | ✅ Optimized | General-purpose | General-purpose | Data pipeline |
| **RDMA throughput** | 270 GB/s | 300+ GB/s | 200+ GB/s | Varies |
| **AI-specific** | Token warehouse | Metadata cache | DGX certified | AIDE engine |
| **Typical nodes** | 16-128 | 8-256 | 8-64 | Varies |
| **Cost** | $200-400K | $300-500K | $250-450K | $200-350K |

### When to Use Each

- **Weka**: Best for KV cache offloading, LMCache integration
- **VAST**: Large-scale AI clusters, general-purpose high-performance
- **Pure Storage**: NVIDIA DGX SuperPOD integration, enterprise SLA
- **NetApp**: Existing NetApp customers, data pipeline focus

---

## Part 5: Integration Architecture

### Current MVP (v0.3.0)

```
Client → FastAPI (8000) → vLLM (8080) → LMCache (local CPU, 64GB)
```

**Limitations:**
- Single-node cache
- No cluster sharing
- Cache lost on restart
- ~100K token limit

### Enterprise Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Tenant Clients                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │Tenant A │  │Tenant B │  │Tenant C │  │Tenant D │  ...       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
        └────────────┼────────────┼────────────┘
                     │            │
                     ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI API Layer (Kubernetes)                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • Tenant isolation middleware                            │    │
│  │ • Authentication/authorization                           │    │
│  │ • Request metering & quota enforcement                   │    │
│  │ • Streaming SSE responses                                │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              vLLM Inference Fleet (8 replicas)                   │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│
│  │vLLM │ │vLLM │ │vLLM │ │vLLM │ │vLLM │ │vLLM │ │vLLM │ │vLLM ││
│  │ #1  │ │ #2  │ │ #3  │ │ #4  │ │ #5  │ │ #6  │ │ #7  │ │ #8  ││
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘│
│     │       │       │       │       │       │       │       │    │
│     └───────┴───────┴───────┼───────┴───────┴───────┴───────┘    │
└─────────────────────────────┼────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LMCache (Weka Backend)                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • Shared KV warehouse (500GB+)                           │    │
│  │ • RDMA-based data access                                 │    │
│  │ • Microsecond latency cache hits                         │    │
│  │ • Distributed across cluster                             │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                Weka Parallel Filesystem                          │
│  • CoreWeave or on-premises                                      │
│  • Multi-tier (cache, warm, cold)                                │
│  • Replication for fault tolerance                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Required Code Modifications

### Phase 1: Configuration (v0.4.0)

**Files:** `src/config.py`, `configs/lmcache.yaml`

```python
# New storage backend options
class StorageBackend(Enum):
    LOCAL_CPU = "local_cpu"
    WEKA = "weka"
    VAST = "vast"
    S3 = "s3"

class LMCacheSettings(BaseSettings):
    backend: StorageBackend = StorageBackend.LOCAL_CPU
    weka_mds_endpoints: Optional[list] = None
    weka_cache_size_gb: int = 64
    weka_enable_gds: bool = True
```

### Phase 2: Multi-Tenancy (v0.4.0)

**Files:** `src/main.py`, `src/middleware/tenancy.py` (new)

```python
@app.middleware("http")
async def extract_tenant(request: Request, call_next):
    tenant_id = request.headers.get("X-Tenant-ID")
    if not tenant_id:
        raise HTTPException(401, "Missing X-Tenant-ID header")

    tenant_ctx = await db.get_tenant(tenant_id)
    request.state.tenant = tenant_ctx

    response = await call_next(request)
    return response
```

### Phase 3: Observability (v0.4.0)

**Files:** `src/metrics.py` (new)

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('inference_requests_total', 'Total requests', ['tenant', 'model'])
REQUEST_LATENCY = Histogram('inference_latency_seconds', 'Request latency', ['tenant', 'model'])
TTFT = Histogram('inference_ttft_seconds', 'Time to first token', ['tenant', 'model'])
ACTIVE_REQUESTS = Gauge('inference_active_requests', 'Currently processing requests')
```

### Phase 4: Kubernetes Deployment (v0.5.0)

**Files:** `kubernetes/deployment.yaml`, `kubernetes/service.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
spec:
  replicas: 8
  template:
    spec:
      containers:
      - name: vllm
        args:
        - "serve"
        - "meta-llama/Llama-3.1-70B"
        - "--tensor-parallel-size=2"
        - "--pipeline-parallel-size=2"
        resources:
          limits:
            nvidia.com/gpu: "2"
        volumeMounts:
        - name: weka-cache
          mountPath: /mnt/weka
```

---

## Part 7: Cost Analysis

### Monthly Costs (24/7 operation)

| Configuration | H100 Cost | Storage | Networking | Total | Per 1M tokens |
|---------------|-----------|---------|------------|-------|---------------|
| **MVP** (1x H100) | $1,430 | $0 | $0 | **$1,430** | $0.012 |
| **Team** (4x H100) | $5,720 | $2,000 | $500 | **$8,220** | $0.008 |
| **Enterprise** (8x H100) | $11,440 | $5,000 | $1,500 | **$17,940** | $0.006 |
| **DGX SuperPOD** (64x H100) | $91,520 | $15,000 | $3,000 | **$109,520** | $0.005 |

### ROI Example

**Assumptions:** 100 customers, 10B tokens/month, $8/$24 per 1M prompt/completion

```
Revenue:  $176K/month
Costs:    $17K/month (8x H100 team scale)
Margin:   90.2%
```

---

## Part 8: Implementation Roadmap

### v0.4.0 - OpenRouter Ready (Current Priority)
- [ ] API key authentication
- [ ] Rate limiting (429 responses)
- [ ] Request logging
- [ ] Prometheus metrics

### v0.5.0 - Enterprise Foundation
- [ ] Multi-tenant middleware
- [ ] Weka storage backend option
- [ ] Kubernetes deployment manifests
- [ ] SOC2 compliance prep

### v0.6.0 - Production Hardening
- [ ] Circuit breaker for vLLM
- [ ] SLA-aware scheduling
- [ ] Distributed inference (tensor/pipeline parallelism)
- [ ] Full compliance certification

### v1.0.0 - Enterprise GA
- [ ] Multi-region deployment
- [ ] On-premises DGX support
- [ ] Advanced billing/metering
- [ ] 99.99% SLA capability

---

## Sources

- [Weka Augmented Memory Grid](https://www.weka.io/blog/ai-ml/wekas-augmented-memory-grid-pioneering-a-token-warehouse-for-the-future/)
- [Weka + NVIDIA Dynamo](https://www.weka.io/blog/ai-ml/weka-accelerates-ai-inference-with-nvidia-dynamo-and-nvidia-nixl/)
- [Weka 75x TTFT Improvement](https://www.weka.io/blog/ai-ml/weka-sets-a-new-bar-with-75x-faster-time-to-first-token-ttft/)
- [LMCache Technical Report](https://arxiv.org/html/2510.09665)
- [vLLM Distributed Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [NVIDIA DGX SuperPOD](https://docs.nvidia.com/dgx-superpod/reference-architecture/scalable-infrastructure-h200/latest/)
- [VAST Data Architecture](https://www.datagravity.dev/p/what-is-vast-data-behind-the-91b)
- [Pure Storage FlashBlade](https://blog.purestorage.com/solutions/pure-storage-nvidia-ai-data-platform/)

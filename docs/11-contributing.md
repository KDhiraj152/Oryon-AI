# Contributing & Project History

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## About the Author

I am K Dhiraj, the architect and primary developer of Oryon AI. This project represents my vision for making AI accessible to multilingual users through AI technology. Every architectural decision, model selection, and optimization reflects lessons learned from building production AI systems at scale.

**Contact:**
- **Email:** k.dhiraj.srihari@gmail.com
- **GitHub:** github.com/kdhiraj152
- **LinkedIn:** www.linkedin.com/in/k-dhiraj

---

## Project Evolution

### Version Timeline

| Version | Date | Focus |
|---------|------|-------|
| 1.0 | Q1 2024 | Initial prototype with basic Q&A |
| 2.0 | Q2 2024 | Multi-language support, RAG implementation |
| 3.0 | Q3 2024 | Production hardening, authentication |
| 4.0 | Q4 2024 | Universal mode, 2025 model stack, M4 optimization |
| 4.1 | Q1 2026 | Documentation restructuring, naming standardization |

---

## Core Contributions

### 1. AI Architecture

**2025 Optimal Model Stack Selection**

Selected and integrated the optimal combination of AI models for the multilingual AI context:

| Component | Model | Rationale |
|-----------|-------|-----------|
| LLM | Qwen3-8B (MLX 4-bit) | Best quality-to-latency ratio at 8B scale, handles both simplification and validation |
| Embeddings | BGE-M3 | Multilingual dense embeddings, SOTA on MTEB |
| Reranker | BGE-Reranker-v2-M3 | Cross-encoder reranking for precision |
| Translation | IndicTrans2-1B | Purpose-built for Indian languages, 22 language support |
| STT | Whisper V3 Turbo | 8x faster than V3, maintains accuracy |
| TTS | Edge-TTS + MMS-TTS | Zero-latency Edge fallback, open-source MMS primary |

**Key Implementation:**
```python
# Model configuration with quantization
MODEL_CONFIGS = {
    "llm": {
        "model_id": "mlx-community/Qwen3-8B-4bit",
        "quantization": "4bit",
        "context_length": 32768,
        "load_in_4bit": True,
    },
    "embedder": {
        "model_id": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "max_seq_length": 8192,
    },
}
```

### 2. Memory Coordination System

Developed a sophisticated memory coordinator for optimal GPU/CPU memory utilization:

```python
class MemoryCoordinator:
    """Coordinates memory across ML models."""

    def __init__(self, max_memory_gb: float = 16.0):
        self.max_memory = max_memory_gb * 1024**3
        self.allocations: dict[str, ModelAllocation] = {}
        self.lock = asyncio.Lock()

    async def allocate(self, model_id: str, required_memory: int) -> bool:
        async with self.lock:
            current_usage = sum(a.memory for a in self.allocations.values())

            if current_usage + required_memory > self.max_memory:
                await self._evict_lru(required_memory)

            self.allocations[model_id] = ModelAllocation(
                model_id=model_id,
                memory=required_memory,
                last_access=time.time(),
            )
            return True

    async def _evict_lru(self, needed: int):
        """Evict least recently used models to free memory."""
        sorted_models = sorted(
            self.allocations.items(),
            key=lambda x: x[1].last_access,
        )

        freed = 0
        for model_id, allocation in sorted_models:
            if freed >= needed:
                break
            await self._unload_model(model_id)
            freed += allocation.memory
            del self.allocations[model_id]
```

### 3. RAG Pipeline

Built a production-grade RAG pipeline with circuit breaker protection:

```python
class RAGService:
    def __init__(self):
        self.embedder = BGEM3Embedder()
        self.reranker = BGEReranker()
        self.vector_store = PGVectorStore()
        self.circuit = CircuitBreaker(failure_threshold=5)

    async def retrieve_and_generate(
        self,
        query: str,
        top_k: int = 5,
    ) -> RAGResponse:
        # Encode query
        query_embedding = await self.embedder.encode_query(query)

        # Vector search
        candidates = await self.vector_store.similarity_search(
            embedding=query_embedding,
            limit=top_k * 3,  # Over-retrieve for reranking
        )

        # Rerank for precision
        reranked = await self.reranker.rerank(
            query=query,
            documents=[c.content for c in candidates],
            top_k=top_k,
        )

        # Generate with context
        context = "\n\n".join([doc for _, doc, _ in reranked])
        response = await self.llm.generate(
            prompt=self._format_prompt(query, context),
        )

        return RAGResponse(
            answer=response,
            sources=[candidates[i] for i, _, _ in reranked],
            confidence=self._calculate_confidence(reranked),
        )
```

### 4. Apple Silicon Optimization

Implemented comprehensive optimizations for M-series chips:

```python
class AppleSiliconOptimizer:
    """Optimize ML workloads for Apple Silicon."""

    @staticmethod
    def configure_torch():
        if torch.backends.mps.is_available():
            # Enable MPS for PyTorch
            torch.set_default_device("mps")

            # Optimize memory allocation
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"

    @staticmethod
    def get_optimal_batch_size(model_size_gb: float) -> int:
        """Calculate optimal batch size based on available memory."""
        available = psutil.virtual_memory().available / 1024**3
        # Reserve 4GB for system, use 60% of remainder
        usable = (available - 4) * 0.6

        # Approximate tokens per GB for inference
        tokens_per_gb = 2048

        return int(usable / model_size_gb * tokens_per_gb)
```

### 5. Multi-language Translation

Integrated IndicTrans2 for authentic Indian language support:

```python
class TranslationService:
    LANGUAGE_CODES = {
        "hindi": "hin_Deva",
        "tamil": "tam_Taml",
        "telugu": "tel_Telu",
        "bengali": "ben_Beng",
        "marathi": "mar_Deva",
        "gujarati": "guj_Gujr",
        "kannada": "kan_Knda",
        "malayalam": "mal_Mlym",
        "punjabi": "pan_Guru",
        "odia": "ory_Orya",
    }

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        source_code = self.LANGUAGE_CODES.get(source_lang, "eng_Latn")
        target_code = self.LANGUAGE_CODES[target_lang]

        # Use IndicTrans2 for Indian languages
        result = await self.indictrans.translate(
            text=text,
            src_lang=source_code,
            tgt_lang=target_code,
        )

        return TranslationResult(
            translated_text=result,
            source_lang=source_lang,
            target_lang=target_lang,
            confidence=self._calculate_confidence(text, result),
        )
```

### 6. Frontend Architecture

Built a responsive React application with Zustand state management:

```typescript
// Zustand store with persistence
export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,

      login: async (credentials) => {
        const response = await api.auth.login(credentials);
        set({
          user: response.user,
          token: response.token,
          isAuthenticated: true,
        });
      },

      logout: () => {
        set({ user: null, token: null, isAuthenticated: false });
      },

      refreshToken: async () => {
        const token = get().token;
        if (!token) return;

        const newToken = await api.auth.refresh(token);
        set({ token: newToken });
      },
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ token: state.token }),
    }
  )
);
```

### 7. Database Architecture

Designed a scalable PostgreSQL schema with pgvector:

```sql
-- Vector embeddings with HNSW index
CREATE TABLE document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    embedding vector(1024),
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_embeddings_hnsw ON document_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Multi-tenant isolation
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE,
    settings JSONB DEFAULT '{}'
);

-- Row-level security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON documents
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
```

### 8. API Design

Designed RESTful APIs with comprehensive documentation:

```python
@router.post(
    "/ask",
    response_model=QAResponse,
    summary="Ask a question",
    description="Submit a question and receive an AI-generated answer",
)
async def ask_question(
    request: QuestionRequest,
    current_user: User = Depends(get_current_user),
    qa_service: QAService = Depends(get_qa_service),
) -> QAResponse:
    """
    Ask a question to the AI assistant.

    - **question**: The question to ask
    - **context_id**: Optional conversation context
    - **language**: Target response language
    - **stream**: Enable streaming response
    """
    response = await qa_service.answer(
        question=request.question,
        user_id=current_user.id,
        language=request.language,
        context_id=request.context_id,
    )

    return QAResponse(
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence,
        language=response.language,
    )
```

---

## Metrics

### Codebase Statistics

| Metric | Count |
|--------|-------|
| Python files | 145 |
| TypeScript files | 67 |
| Lines of code | 28,000+ |
| Test files | 48 |
| Test coverage | 90% |

### Performance Benchmarks

| Metric | Result |
|--------|--------|
| Question answering latency (p50) | 450ms |
| Question answering latency (p95) | 850ms |
| Translation latency | 120ms |
| Speech synthesis latency | 200ms |
| Concurrent users supported | 10,000+ |

### Model Performance

| Model | Accuracy/Quality |
|-------|-----------------|
| RAG retrieval (MRR@10) | 0.87 |
| Reranking improvement | +12% |
| Translation BLEU (Hindi) | 42.3 |
| STT word error rate | 8.2% |

---

## Key Decisions

### 1. Qwen3-8B over Qwen2.5-3B + Gemma-2-2B

Selected Qwen3-8B as a unified LLM to replace both Qwen2.5-3B-Instruct and Gemma-2-2B-IT:
- Superior multilingual performance, especially for transliterated text
- Better instruction following in content contexts
- More efficient INT4 quantization with minimal quality loss
- Longer context window (32K vs 8K)

### 2. BGE-M3 over OpenAI Embeddings

Chose BGE-M3 for embeddings instead of external APIs:
- Zero latency cost from API calls
- Privacy-compliant (all data stays local)
- Multilingual support without additional models
- 1024-dimension vectors with excellent retrieval quality

### 3. PostgreSQL + pgvector over Pinecone

Selected self-hosted pgvector over managed vector databases:
- Unified database for relational and vector data
- No vendor lock-in or per-query costs
- HNSW indexes match or exceed managed performance
- Full SQL capabilities for complex queries

### 4. Edge-TTS as Fallback

Implemented dual TTS strategy:
- MMS-TTS as primary (open-source, no dependencies)
- Edge-TTS as fallback (higher quality, requires connectivity)
- Automatic failover with circuit breaker pattern

---

## Codebase Restructuring (June 2025)

Full structural overhaul across backend, frontend, scripts, and documentation. This restructuring addressed architectural debt accumulated during rapid feature development. The codebase was standardized without altering runtime behavior.

### Dead Code Elimination

| File / Pattern | Action | Reason |
|---|---|---|
| `backend/integration.py` | Deleted | Zero consumers; legacy sync-only code (521 lines) |
| `frontend/src/components/OmLogo.tsx` | Deleted | Dead; all imports use `landing/OmLogo.tsx` |
| `ReviewQueue` global in `services/__init__.py` | Removed | Dead singleton pattern, never instantiated |

### Type Unification (4x ModelConfig → 1 Canonical Source)

Created `backend/core/types.py` as the single source of truth for shared enums:
- `ModelTier` (FAST / BALANCED / QUALITY / PREMIUM)
- `TaskType` (CHAT / REASONING / CODE / TRANSLATION / EMBEDDING)
- `ModelType` (LLM / EMBEDDING / TTS / STT / OCR / TRANSLATION / RERANKER)

### CircuitBreaker Consolidation (3x → 1)

| Location | Before | After |
|---|---|---|
| `core/circuit_breaker.py` | Canonical implementation | Unchanged (canonical) |
| `core/exceptions.py` | ~100-line duplicate class | Re-exports from `circuit_breaker.py` |
| `pipeline/orchestrator_v2.py` | `PipelineCircuitBreaker` | Cleaned up (proper docs, private attrs) |

### Lazy-Loading `core/optimized/`

Rewrote `backend/core/optimized/__init__.py` from 384-line eager imports of 21 submodules to a ~220-line lazy-loading system using `__getattr__` + `_LAZY_IMPORTS` registry (180+ symbols). Reduces cold-start import time significantly.

### Script Reorganization

- **Extensions fixed** — Python scripts with `.sh` extensions corrected to `.py`
- **Naming normalized** — kebab-case → snake_case across all scripts
- **Testing consolidated** — 15 → 11 scripts (removed duplicates, merged overlaps)

### Key Fixes

- **AgentRegistry singleton**: Fixed `__new__` + `_initialized` guard so all callers share one registry
- **Security gap in `system.ts`**: Replaced direct `localStorage` token reads with secure closure-based manager
- **GPU coordination extracted**: ~120 lines moved from `inference/__init__.py` to dedicated `gpu_coordination.py`

### Backward Compatibility

All renamed types have backward-compatible aliases:
- `ModelConfig = HardwareModelConfig` (in `model_manager.py`)
- `TaskType = DeviceTaskType` (in `device_router.py`)
- `CoreMLInferenceEngine = CoreMLEmbeddingEngine` (in `inference/__init__.py`)
- `WarmupService = ModelWarmupService` (in `inference/__init__.py`)

---

## Acknowledgments

This project builds on the work of many open-source contributors:

- **Hugging Face** — Model hosting and transformers library
- **BAAI** — BGE embedding and reranking models
- **AI4Bharat** — IndicTrans2 translation models
- **Qwen Team** — Qwen3 language models
- **OpenAI** — Whisper speech recognition
- **Meta** — MMS text-to-speech
- **PostgreSQL** — Database foundation
- **pgvector** — Vector similarity search extension

---

## Contact

For questions, collaboration opportunities, or feedback:

**K Dhiraj**
k.dhiraj.srihari@gmail.com

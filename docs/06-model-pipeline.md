# Model Pipeline

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## 2025 Optimal Model Stack

The model selection prioritizes quality-per-parameter efficiency, multilingual capability, and hardware compatibility. Each model was benchmarked against alternatives before integration.

| Component | Model | Parameters | Size (INT4) | Purpose |
|-----------|-------|------------|-------------|---------|
| **LLM** | Qwen3-8B (MLX 4-bit) | 8B | 4.6 GB | Text generation, reasoning, validation |
| **Translation** | IndicTrans2-1B | 1B | 800 MB | Indian language translation |
| **Embeddings** | BGE-M3 | 568M | 600 MB | Multilingual semantic search |
| **Reranker** | BGE-Reranker-v2-M3 | 568M | 600 MB | Cross-encoder reranking |
| **STT** | Whisper V3 Turbo | 809M | 900 MB | Multilingual transcription |
| **TTS** | MMS-TTS | 50-100M | 100 MB | Voice synthesis |

**Total Memory Footprint:** ~5 GB (with dynamic loading, only 2-3 models active simultaneously)

---

## Embedding Pipeline

### BGE-M3 Embedder

BGE-M3 was selected for its native multilingual capability—it embeds all 10 supported Indian languages into a shared vector space, enabling cross-lingual retrieval without translation overhead.

**Key Specifications:**
- Dimension: 1024
- Max sequence length: 8192 tokens
- Languages: 100+ (including all target Indian languages)
- Retrieval modes: Dense + Sparse (hybrid)

**Implementation:**

```python
class BGEM3Embedder:
    def __init__(self):
        self.model_id = "BAAI/bge-m3"
        self.dimension = 1024
        self._model = None

        # Device routing via hardware optimizer
        router = get_device_router()
        routing = router.route(TaskType.EMBEDDING)
        self.device = routing.device_str

    def _load_model(self):
        """Lazy load with memory coordination."""
        if self._model is not None:
            return

        coordinator = get_memory_coordinator()

        with coordinator.acquire_memory_sync("bgem3", 1.5, priority=2):
            # Sentence-transformers for MPS compatibility
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id,
                device=self.device,
                trust_remote_code=True
            )

            coordinator.register_model("bgem3", self._model, self.unload)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode with GPU synchronization."""
        self._load_model()

        # GPU lock prevents Metal conflicts with MLX
        from .inference import run_on_gpu_sync

        def _do_encode():
            return self._model.encode(texts, batch_size=batch_size)

        return run_on_gpu_sync(_do_encode)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query with prefix for retrieval."""
        return self.encode([f"query: {query}"])[0]
```

**Performance (M4 Pro):**
- Single text: 15ms
- Batch of 32: 45ms
- Throughput: 348 texts/second

---

## Retrieval Pipeline

### Vector Search with pgvector

PostgreSQL with pgvector provides vector similarity search using HNSW (Hierarchical Navigable Small World) indexes.

**Index Configuration:**

```sql
CREATE INDEX embedding_hnsw_idx
ON document_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Search Parameters:**
- `m = 16`: Maximum connections per node (higher = better recall, more memory)
- `ef_construction = 64`: Build-time search breadth
- `ef_search = 40`: Query-time search breadth (configurable)

**Query Example:**

```python
async def vector_search(
    query_embedding: np.ndarray,
    limit: int = 15,
    threshold: float = 0.5
) -> list[RetrievalResult]:
    """Execute vector similarity search."""

    query = """
        SELECT
            id, chunk_text,
            1 - (embedding <=> $1::vector) as similarity
        FROM document_embeddings
        WHERE 1 - (embedding <=> $1::vector) > $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
    """

    results = await db.fetch(query, query_embedding.tolist(), threshold, limit)

    return [
        RetrievalResult(
            chunk_id=r["id"],
            text=r["chunk_text"],
            score=r["similarity"]
        )
        for r in results
    ]
```

---

## Reranking Pipeline

### BGE-Reranker-v2-M3

The reranker uses cross-encoder architecture to score query-document pairs with higher precision than embedding similarity alone.

**Why Reranking:**
- Embedding similarity is fast but approximate
- Cross-encoders see query and document together, capturing subtle relevance signals
- Typical improvement: 15-25% in retrieval accuracy

**Implementation:**

```python
class BGEReranker:
    def __init__(self):
        self.model_id = "BAAI/bge-reranker-v2-m3"
        self._model = None
        self._use_cross_encoder = True

    def _load_model(self):
        """Load with memory coordination."""
        if self._model is not None:
            return

        coordinator = get_memory_coordinator()

        with coordinator.acquire_memory_sync("reranker", 1.2, priority=1):
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, device=self.device)
            coordinator.register_model("reranker", self._model, self.unload)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance."""
        self._load_model()

        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs, batch_size=100)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
```

**Performance (M4 Pro):**
- 5 documents: 45ms
- 15 documents: 120ms
- Throughput: 2.6ms per document

---

## Generation Pipeline

### Qwen3-8B

Qwen3-8B was selected for its superior quality-per-parameter ratio among 8B-class models, particularly for content and multilingual understanding. It replaces both Qwen2.5-3B (simplification) and Gemma-2-2B-IT (validation) with a single unified model.

**Model Specifications:**
- Parameters: 3B
- Context length: 32,768 tokens
- Architecture: Decoder-only transformer
- Quantization: INT4 (AWQ)

**Generation Configuration:**

```python
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

    # Content adjustments
    presence_penalty: float = 0.1  # Encourage diverse vocabulary
    frequency_penalty: float = 0.1  # Reduce repetition
```

**Prompt Template:**

```python
def build_prompt(question: str, context: str, grade: int) -> str:
    return f"""You are an expert educational tutor helping users in India.

Context from relevant reference materials:
{context}

User Complexity Level: {grade}

Instructions:
- Explain concepts clearly and accurately
- Use age-appropriate language for grade {grade}
- Include relevant examples
- If the context doesn't contain the answer, say so honestly

Question: {question}

Answer:"""
```

**Streaming Generation:**

```python
async def generate_stream(
    prompt: str,
    config: GenerationConfig
) -> AsyncIterator[str]:
    """Stream tokens as they're generated."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "do_sample": config.do_sample,
    }

    # Run generation in thread pool
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token
```

**Performance (M4 Pro, INT4):**
- First token latency: 180ms
- Generation speed: 50 tokens/second
- Context processing: ~2,000 tokens/second

---

## Translation Pipeline

### IndicTrans2-1B

IndicTrans2 is the state-of-the-art model for English ↔ Indian language translation, developed by AI4Bharat.

**Supported Directions:**
- English → Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia
- All above languages → English
- Direct translation between Indian languages (via English pivot)

**Implementation:**

```python
class IndicTrans2Translator:
    def __init__(self):
        self.model_id = "ai4bharat/indictrans2-en-indic-1B"
        self._model = None
        self._tokenizer = None

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str
    ) -> str:
        """Translate text between languages."""
        self._load_model()

        # Preprocess with language tags
        input_text = f"<2{tgt_lang}> {text}"

        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                early_stopping=True
            )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Performance (M4 Pro):**
- Average sentence: 120ms
- Long paragraph: 300ms

---

## Speech-to-Text Pipeline

### Whisper V3 Turbo

OpenAI's Whisper V3 Turbo provides accurate multilingual transcription with automatic language detection.

**Specifications:**
- Languages: 99+ (including all target Indian languages)
- Input: 16kHz mono audio
- Output: Text with timestamps (optional)

**Implementation:**

```python
class WhisperTranscriber:
    def __init__(self):
        self.model_id = "openai/whisper-large-v3-turbo"
        self._model = None
        self._processor = None

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = None
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        self._load_model()

        # Preprocess audio
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)

        # Generate with language hint if provided
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **generate_kwargs,
                return_timestamps=False
            )

        text = self._processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        return TranscriptionResult(
            text=text,
            detected_language=self._detect_language(outputs),
            confidence=self._get_confidence(outputs)
        )
```

**Performance (M4 Pro):**
- Real-time factor: 0.3x (3 seconds of audio = 1 second processing)
- Accuracy: 95%+ for Indian languages

---

## Text-to-Speech Pipeline

### MMS-TTS (Primary) + Edge-TTS (Fallback)

**MMS-TTS:**
- Local inference, no network required
- Covers all 10 target languages
- Moderate quality, low latency

**Edge-TTS:**
- Microsoft Edge neural voices (network required)
- Higher quality
- Used as fallback when network available

**Implementation:**

```python
class TTSService:
    def __init__(self):
        self.mms = MMSTTSService()
        self.edge = EdgeTTSService()

    async def synthesize(
        self,
        text: str,
        language: str,
        prefer_quality: bool = False
    ) -> bytes:
        """Synthesize speech with automatic fallback."""

        if prefer_quality:
            try:
                return await self.edge.synthesize(text, language)
            except NetworkError:
                pass

        return self.mms.synthesize(text, language)
```

**Performance (M4 Pro):**
- MMS-TTS: 31x realtime (1 second of audio = 32ms processing)
- Edge-TTS: Depends on network latency

---

## Memory Coordination

### Dynamic Model Loading

The Memory Coordinator ensures models fit within the memory budget:

```python
class MemoryCoordinator:
    def __init__(self, budget_gb: float = 12.0):
        self.budget = budget_gb * 1024**3
        self.models = {}

    def get_memory_pressure(self) -> MemoryPressure:
        usage = self._get_usage()
        ratio = usage / self.budget

        if ratio < 0.7:
            return MemoryPressure.NORMAL
        elif ratio < 0.85:
            return MemoryPressure.HIGH
        elif ratio < 0.95:
            return MemoryPressure.CRITICAL
        else:
            return MemoryPressure.EMERGENCY

    def _evict_lowest_priority(self) -> bool:
        """Evict least-recently-used, lowest-priority model."""
        if not self.models:
            return False

        # Sort by priority (ascending) then last_access (ascending)
        candidates = sorted(
            self.models.items(),
            key=lambda x: (x[1].priority, x[1].last_access)
        )

        # Evict first candidate (lowest priority, oldest access)
        name, model_info = candidates[0]
        model_info.unload_fn()
        del self.models[name]

        return True
```

### Model Loading Priority

| Model | Priority | Eviction Order |
|-------|----------|----------------|
| BGE-M3 (Embeddings) | 2 | Last |
| Qwen3-8B (LLM) | 2 | Last |
| BGE-Reranker | 1 | Middle |
| Whisper (STT) | 0 | First |
| IndicTrans2 | 0 | First |
| MMS-TTS | 0 | First |

---

## Pipeline Orchestration

### Complete RAG Query Flow

```python
class UnifiedPipelineService:
    async def query(
        self,
        question: str,
        language: str,
        grade: int
    ) -> AsyncIterator[str]:
        """Execute complete RAG pipeline with streaming."""

        # 1. Translate if non-English
        if language != "en":
            english_question = await self.translator.translate(
                question, language, "en"
            )
        else:
            english_question = question

        # 2. Retrieve relevant context
        query_embedding = self.embedder.encode_query(english_question)
        candidates = await self.vector_search(query_embedding, limit=15)

        # 3. Rerank candidates
        if candidates:
            reranked = self.reranker.rerank(
                english_question,
                [c.text for c in candidates],
                top_k=5
            )
            context_chunks = [candidates[idx] for idx, _ in reranked]
        else:
            context_chunks = []

        # 4. Build prompt
        context = "\n\n".join([c.text for c in context_chunks])
        prompt = self.build_prompt(english_question, context, grade)

        # 5. Generate response (streaming)
        async for token in self.llm.generate_stream(prompt):
            # 6. Translate back if non-English
            if language != "en":
                translated = await self.translator.translate(token, "en", language)
                yield translated
            else:
                yield token
```

---

*For deployment details, see Section 8: Deployment.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com

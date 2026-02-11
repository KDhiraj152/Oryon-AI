# Executive Summary

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## Project Overview

**Shiksha Setu v4.0** is a production-grade AI education platform engineered to deliver world-class AI capabilities to every user. This is not a wrapper around existing APIs—it is a complete, vertically integrated system designed from first principles to solve real problems in multilingual AI.

The platform achieves what cloud-based solutions cannot: **full local deployment with zero data transmission, native multilingual support across 10 Indian languages, and hardware-optimized inference that runs on consumer devices.**

The architecture prioritizes three non-negotiables:
1. **Privacy**: All AI processing occurs on-device. No user data leaves the local network.
2. **Accessibility**: Works offline after initial setup. No subscriptions, no metered usage.
3. **Quality**: Domain-aware responses that match content standards across complexity levels.

---

## The Market Gap

Three systemic failures define the current EdTech landscape in India:

### 1. Language Exclusion

India has 22 officially recognized languages. The overwhelming majority of quality content AI operates exclusively in English, creating an artificial barrier for 800+ million non-English-speaking users. A student in rural Tamil Nadu has the same intellectual capacity as one in Mumbai—they simply lack access to tools that understand their language.

Shiksha Setu integrates **IndicTrans2**, the state-of-the-art Indian language translation model, directly into the inference pipeline. The system provides native-quality responses in Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, and Odia.

### 2. Infrastructure Constraints

Cloud-based AI solutions require consistent high-speed internet and charge per-API-call fees. This model is fundamentally incompatible with Tier 2/3 cities and rural areas where connectivity is unreliable and cost-prohibitive.

Shiksha Setu eliminates this dependency entirely. After a one-time ~10GB model download, the system operates fully offline. A user in a remote area runs the same powerful AI stack as urban institutions.

### 3. Pedagogical Misalignment

General-purpose language models produce technically accurate but pedagogically inappropriate responses. They lack understanding of content structure, complexity-appropriate complexity, and the specific context users require.

Shiksha Setu is purpose-built for AI. The system understands complexity levels, aligns with content domain standards, and adapts explanations based on the student's learning profile.

---

## Technical Architecture

The platform combines multiple specialized AI models with custom infrastructure optimized for consumer hardware:

### Core Model Stack (2025 Optimal Configuration)

| Component | Model | Purpose |
|-----------|-------|---------|
| **Reasoning** | Qwen3-8B (MLX 4-bit) | Text generation, explanation, Q&A, validation |
| **Translation** | IndicTrans2-1B | Indian language translation |
| **Embeddings** | BGE-M3 | Multilingual semantic search |
| **Reranking** | BGE-Reranker-v2-M3 | Retrieval accuracy optimization |
| **Speech-to-Text** | Whisper V3 Turbo | Multilingual transcription |
| **Text-to-Speech** | MMS-TTS | Voice synthesis across languages |

### Hardware Optimization

Primary optimization target is Apple Silicon M4, with full support for NVIDIA CUDA and CPU-only operation:

- **Global Memory Coordinator**: Orchestrates 6+ AI models in unified memory with LRU eviction, thermal monitoring, and memory pressure detection
- **Dynamic Model Loading**: Models load on-demand and unload under memory pressure
- **Device-Aware Routing**: Operations route to optimal compute units (GPU, Neural Engine, CPU cores) based on real-time conditions

### Retrieval-Augmented Generation

The RAG implementation goes beyond basic vector search:

- **Hybrid Retrieval**: BGE-M3 provides both dense (semantic) and sparse (keyword) retrieval simultaneously
- **Cross-Encoder Reranking**: BGE-Reranker-v2-M3 scores relevance with precision
- **Semantic Validation**: Filters retrieved chunks that don't actually answer the question, reducing hallucinations
- **Self-Optimization**: Retrieval parameters adjust based on user interaction patterns

---

## Differentiating Architecture Decisions

### Universal Mode with Intelligent Safety

The platform implements **Universal Mode**—a configuration that enables unrestricted open exploration while maintaining genuine safety. The 3-Pass Safety Pipeline:

1. **Semantic Pass**: Analyzes query intent using embedding similarity
2. **Logical Pass**: Evaluates potential for real-world harm
3. **Policy Pass**: Applies configurable policies for different deployment contexts

This approach enables academic discussion of complex topics while blocking genuinely harmful requests.

### Predictive Resource Scheduling

The GPU Resource Scheduler makes dynamic decisions based on real-time conditions:

- **Thermal Monitoring**: Routes operations away from hot compute units
- **Memory Pressure Detection**: Evicts cold models before loading new ones
- **Device Capability Matching**: Embeddings route to Neural Engine; attention operations route to GPU
- **Batch Size Adaptation**: Adjusts based on available memory

### Privacy by Design

This is not a feature—it is an architectural constraint. There are no external API calls, no telemetry, no analytics pipelines. Users can ask questions freely without surveillance, struggle with concepts without performance tracking, and explore topics without data collection.

---

## Performance Metrics

Benchmarked on Apple Silicon M4 Pro with 16GB unified memory:

| Operation | Performance | Notes |
|-----------|-------------|-------|
| **Embedding Throughput** | 348 texts/sec | BGE-M3 with MLX optimization |
| **LLM Inference** | 50 tokens/sec | Qwen3-8B with MLX 4-bit quantization |
| **Text-to-Speech** | 31x realtime | MMS-TTS on Apple Silicon |
| **Speech-to-Text** | 2x realtime | Whisper V3 Turbo |
| **Reranking Latency** | 2.6ms/document | BGE-Reranker-v2-M3 |
| **SIMD Throughput** | 54.7M ops/sec | Cosine similarity operations |
| **Memory Efficiency** | 75% reduction | INT4 quantization vs FP16 |

End-to-end latency for a voice-to-voice query (transcription → RAG → generation → synthesis): **under 4 seconds on consumer hardware.**

---

## Strategic Positioning

Shiksha Setu occupies a unique position in the market:

| Capability | Cloud Solutions | Shiksha Setu |
|------------|-----------------|--------------|
| Data Privacy | Data transmitted to foreign servers | All processing local |
| Offline Operation | Requires internet | Full offline capability |
| Language Support | English-first, translation as afterthought | Native multilingual from ground up |
| Cost Structure | Subscription/per-call | One-time setup, zero ongoing cost |
| Curriculum Alignment | Generic responses | domain-aligned, grade-aware |

---

*For technical implementation details, refer to the subsequent architecture documentation.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com

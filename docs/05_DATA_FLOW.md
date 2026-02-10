# Section 5: Data Flow

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.0.0 (Universal Mode)
**Last Updated:** December 5, 2025

---

## Overview

This document traces the complete data flow through Shiksha Setu for the primary use cases: Question-Answering, Voice Interaction, and Document Processing. Each trace shows the exact path data takes from user input to final response.

---

## Use Case 1: Text Question-Answering

### Request Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        TEXT Q&A DATA FLOW                                        │
└──────────────────────────────────────────────────────────────────────────────────┘

User types: "न्यूटन का पहला नियम क्या है?" (What is Newton's First Law? - in Hindi)

┌─────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Browser   │────▶│  Zustand    │────▶│  API Client     │────▶│   FastAPI   │
│   Input     │     │  Store      │     │  (Axios)        │     │   /qa/ask   │
└─────────────┘     └─────────────┘     └─────────────────┘     └──────┬──────┘
                                                                        │
                    ┌───────────────────────────────────────────────────┘
                    ▼
           ┌────────────────┐
           │   MIDDLEWARE   │
           │                │
           │ 1. JWT Auth    │
           │ 2. Rate Limit  │
           │ 3. Request ID  │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │ LANGUAGE       │
           │ DETECTION      │
           │                │
           │ Input: Hindi   │
           │ Script: Devanagari │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  TRANSLATION   │────▶│   IndicTrans2-1B   │
           │  (if needed)   │     │   Hindi → English  │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ Translated: "What is Newton's First Law?"
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  EMBEDDING     │────▶│     BGE-M3         │
           │  GENERATION    │     │   1024-dim vector  │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ Query vector: [0.023, -0.156, ...]
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  VECTOR        │────▶│   PostgreSQL +     │
           │  SEARCH        │     │   pgvector HNSW    │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ Top-15 candidate chunks
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  RERANKING     │────▶│   BGE-Reranker     │
           │                │     │   Cross-Encoder    │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ Top-5 reranked chunks with scores
                   ▼
           ┌────────────────┐
           │  CONTEXT       │
           │  ASSEMBLY      │
           │                │
           │  Build prompt  │
           │  with sources  │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  LLM           │────▶│   Qwen2.5-3B       │
           │  GENERATION    │     │   Streaming output │
           └───────┬────────┘     └────────────────────┘
                   │
                   │ English response (streaming)
                   ▼
           ┌────────────────┐     ┌────────────────────┐
           │  TRANSLATION   │────▶│   IndicTrans2-1B   │
           │  (back to Hindi) │   │   English → Hindi  │
           └───────┬────────┘     └────────────────────┘
                   │
                   ▼
           ┌────────────────┐     ┌─────────────┐     ┌─────────────┐
           │  SSE STREAM    │────▶│  React SSE  │────▶│   Browser   │
           │  Response      │     │  Handler    │     │   Display   │
           └────────────────┘     └─────────────┘     └─────────────┘
```

### Timing Breakdown (M4 Pro, 16GB)

| Stage | Latency | Cumulative |
|-------|---------|------------|
| Language Detection | 5ms | 5ms |
| Translation (Hindi→English) | 120ms | 125ms |
| Embedding Generation | 15ms | 140ms |
| Vector Search (HNSW) | 8ms | 148ms |
| Reranking (5 docs) | 45ms | 193ms |
| Context Assembly | 3ms | 196ms |
| LLM First Token | 180ms | 376ms |
| LLM Full Generation | 800ms | 1,176ms |
| Translation (English→Hindi) | 120ms | 1,296ms |
| **Total** | | **~1.3 seconds** |

---

## Use Case 2: Voice Question-Answering

### Request Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       VOICE Q&A DATA FLOW                                        │
└──────────────────────────────────────────────────────────────────────────────────┘

User speaks: "गुरुत्वाकर्षण बल क्या है?" (What is gravitational force? - in Hindi)

┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Microphone │────▶│  MediaRecorder  │────▶│  Audio Blob     │
│  Input      │     │  (WebM/Opus)    │     │  (Binary)       │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                      │
                                              POST /voice/transcribe
                                                      │
                                                      ▼
                                             ┌────────────────┐
                                             │   FastAPI      │
                                             │   Audio Upload │
                                             └───────┬────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐
                                             │  AUDIO         │
                                             │  PREPROCESSING │
                                             │                │
                                             │  • Format: WAV │
                                             │  • Sample: 16kHz│
                                             │  • Channels: 1 │
                                             └───────┬────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐     ┌──────────────────┐
                                             │  SPEECH-TO-    │────▶│  Whisper V3      │
                                             │  TEXT          │     │  Turbo           │
                                             │                │     │  (Multilingual)  │
                                             │  Auto-detect:  │     └──────────────────┘
                                             │  Language: hi  │
                                             └───────┬────────┘
                                                     │
                                                     │ Transcript: "गुरुत्वाकर्षण बल क्या है?"
                                                     │ Detected Language: Hindi
                                                     │
                                                     ▼
                                         ┌────────────────────────┐
                                         │                        │
                                         │   RAG PIPELINE         │
                                         │   (Same as Text Q&A)   │
                                         │                        │
                                         │   Translation →        │
                                         │   Embedding →          │
                                         │   Retrieval →          │
                                         │   Reranking →          │
                                         │   Generation →         │
                                         │   Translation          │
                                         │                        │
                                         └───────────┬────────────┘
                                                     │
                                                     │ Hindi text response
                                                     ▼
                                             ┌────────────────┐     ┌──────────────────┐
                                             │  TEXT-TO-      │────▶│  MMS-TTS         │
                                             │  SPEECH        │     │  (Hindi voice)   │
                                             │                │     │                  │
                                             │  Streaming     │     │  OR Edge-TTS     │
                                             │  audio chunks  │     │  (fallback)      │
                                             └───────┬────────┘     └──────────────────┘
                                                     │
                                                     │ Audio stream (MP3)
                                                     ▼
                                             ┌────────────────┐     ┌─────────────────┐
                                             │  HTTP Response │────▶│  Web Audio API  │
                                             │  (audio/mpeg)  │     │  Playback       │
                                             └────────────────┘     └─────────────────┘
```

### Timing Breakdown (M4 Pro, 16GB)

| Stage | Latency | Cumulative |
|-------|---------|------------|
| Audio Upload | 50ms | 50ms |
| Preprocessing | 20ms | 70ms |
| STT (Whisper V3) | 400ms | 470ms |
| RAG Pipeline | 1,200ms | 1,670ms |
| TTS Generation | 600ms | 2,270ms |
| Audio Streaming | 100ms | 2,370ms |
| **Total** | | **~2.4 seconds** |

---

## Use Case 3: Document Upload and Processing

### Request Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     DOCUMENT PROCESSING DATA FLOW                                │
└──────────────────────────────────────────────────────────────────────────────────┘

User uploads: "NCERT_Physics_Class11_Chapter3.pdf" (2.5MB)

┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  File       │────▶│  FormData       │────▶│  FastAPI        │
│  Picker     │     │  Upload         │     │  /documents/    │
└─────────────┘     └─────────────────┘     └───────┬─────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐
                                             │  FILE          │
                                             │  VALIDATION    │
                                             │                │
                                             │  • Type: PDF   │
                                             │  • Size: 2.5MB │
                                             │  • Virus scan  │
                                             └───────┬────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐
                                             │  STORAGE       │
                                             │                │
                                             │  Save to:      │
                                             │  storage/      │
                                             │  uploads/      │
                                             └───────┬────────┘
                                                     │
                                          ┌──────────┴──────────┐
                                          ▼                     ▼
                                   ┌────────────┐        ┌────────────┐
                                   │  PDF TEXT  │        │  OCR       │
                                   │  EXTRACTION│        │  (if scan) │
                                   │            │        │            │
                                   │  PyMuPDF   │        │  OCRService│
                                   └─────┬──────┘        └─────┬──────┘
                                         │                     │
                                         └──────────┬──────────┘
                                                    │
                                                    │ Raw text (50 pages)
                                                    ▼
                                             ┌────────────────┐
                                             │  TEXT          │
                                             │  CHUNKING      │
                                             │                │
                                             │  • Chunk: 512  │
                                             │  • Overlap: 50 │
                                             │  • Total: 245  │
                                             └───────┬────────┘
                                                     │
                                                     │ 245 text chunks
                                                     ▼
                                             ┌────────────────┐     ┌──────────────────┐
                                             │  BATCH         │────▶│  BGE-M3          │
                                             │  EMBEDDING     │     │  batch_size=32   │
                                             │                │     │                  │
                                             │  8 batches     │     │  Memory-coordinated│
                                             └───────┬────────┘     └──────────────────┘
                                                     │
                                                     │ 245 × 1024-dim vectors
                                                     ▼
                                             ┌────────────────┐     ┌──────────────────┐
                                             │  VECTOR        │────▶│  PostgreSQL      │
                                             │  STORAGE       │     │  pgvector        │
                                             │                │     │                  │
                                             │  Bulk insert   │     │  HNSW index      │
                                             │  with metadata │     │  update          │
                                             └───────┬────────┘     └──────────────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐
                                             │  DATABASE      │
                                             │  RECORD        │
                                             │                │
                                             │  documents:    │
                                             │  • id          │
                                             │  • filename    │
                                             │  • chunk_count │
                                             │  • created_at  │
                                             └───────┬────────┘
                                                     │
                                                     ▼
                                             ┌────────────────┐     ┌─────────────────┐
                                             │  JSON Response │────▶│  UI Update      │
                                             │  {success, id} │     │  Document list  │
                                             └────────────────┘     └─────────────────┘
```

### Timing Breakdown (M4 Pro, 16GB)

| Stage | Latency | Cumulative |
|-------|---------|------------|
| Upload & Validation | 200ms | 200ms |
| Text Extraction | 800ms | 1,000ms |
| Chunking | 50ms | 1,050ms |
| Embedding (245 chunks) | 1,800ms | 2,850ms |
| Vector Storage | 400ms | 3,250ms |
| Index Update | 150ms | 3,400ms |
| **Total** | | **~3.4 seconds** |

---

## Data Flow: Streaming Response

### SSE Token Streaming

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     SSE STREAMING DATA FLOW                                      │
└──────────────────────────────────────────────────────────────────────────────────┘

LLM generates token-by-token:

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Qwen2.5-3B     │────▶│  Token Buffer   │────▶│  SSE Encoder    │
│  generate()     │     │  (5 tokens)     │     │  event: message │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
Token stream: "Newton" → "'s" → " First" → " Law" → ...   │
                                                          ▼
                                                 ┌────────────────┐
                                                 │  HTTP Stream   │
                                                 │                │
                                                 │  data: Newton  │
                                                 │  data: 's      │
                                                 │  data:  First  │
                                                 │  data:  Law    │
                                                 │  ...           │
                                                 │  data: [DONE]  │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  EventSource   │
                                                 │  (Browser)     │
                                                 │                │
                                                 │  onmessage()   │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  Zustand       │
                                                 │  appendChunk() │
                                                 │                │
                                                 │  Real-time UI  │
                                                 │  update        │
                                                 └────────────────┘
```

---

## Error Handling Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     ERROR HANDLING DATA FLOW                                     │
└──────────────────────────────────────────────────────────────────────────────────┘

Error occurs during LLM inference:

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  LLM Inference  │──X──│  CUDA OOM       │────▶│  Circuit        │
│  Exception      │     │  Error          │     │  Breaker        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                 Records failure
                                                 (5 failures = trip)
                                                          │
                                                          ▼
                                                 ┌────────────────┐
                                                 │  Memory        │
                                                 │  Coordinator   │
                                                 │                │
                                                 │  Evict models  │
                                                 │  Clear cache   │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  Exception     │
                                                 │  Handler       │
                                                 │                │
                                                 │  Format error  │
                                                 │  for client    │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  HTTP 503      │
                                                 │                │
                                                 │  {             │
                                                 │    error:      │
                                                 │    "model_     │
                                                 │     overload", │
                                                 │    retry_after:│
                                                 │    30          │
                                                 │  }             │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  React Error   │
                                                 │  Boundary      │
                                                 │                │
                                                 │  Show retry    │
                                                 │  message       │
                                                 └────────────────┘
```

---

## Cache Layer Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-TIER CACHE DATA FLOW                                   │
└──────────────────────────────────────────────────────────────────────────────────┘

Embedding lookup request:

┌─────────────────┐
│  Cache.get()    │
│  key: "hash123" │
└────────┬────────┘
         │
         ▼
┌────────────────────┐     ┌─────────────────┐
│  L1: Memory        │────▶│  HIT?           │──── YES ──▶ Return (1ms)
│  (LRU Dict)        │     │                 │
└────────────────────┘     └────────┬────────┘
                                    │ MISS
                                    ▼
                           ┌────────────────────┐     ┌─────────────────┐
                           │  L2: Redis         │────▶│  HIT?           │──── YES ──▶ Return (5ms)
                           │  (msgpack)         │     │                 │            + Populate L1
                           └────────────────────┘     └────────┬────────┘
                                                               │ MISS
                                                               ▼
                                                      ┌────────────────────┐
                                                      │  L3: Disk          │
                                                      │  (msgpack files)   │
                                                      └────────┬───────────┘
                                                               │
                                                      ┌────────┴────────┐
                                                      ▼                 ▼
                                               HIT (20ms)         MISS (Compute)
                                               Populate L1+L2     Generate embedding
                                                                  Populate all tiers
```

---

*For API endpoint details, see Section 6: API Documentation.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com

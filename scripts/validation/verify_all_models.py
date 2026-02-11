#!/usr/bin/env python3
"""Verify all 7 AI models are cached and loadable."""

import gc
import time
import sys

results = {}


def test_model(name, num, total, test_fn):
    """Run a model test and record result."""
    print("━" * 60)
    print(f"{num}/{total}  {name}")
    print("━" * 60)
    try:
        result = test_fn()
        results[name] = "✅"
        print(f"  ✅ {result}")
    except Exception as e:
        results[name] = f"❌ {str(e)[:80]}"
        print(f"  ❌ ERROR: {e}")
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    print()


def test_qwen3():
    from mlx_lm import load, generate

    t0 = time.perf_counter()
    model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = generate(
        model,
        tokenizer,
        prompt="Simplify for grade 5: Photosynthesis is a biochemical process.",
        max_tokens=50,
    )
    gen_time = time.perf_counter() - t0
    del model, tokenizer
    return f"Load: {load_time:.1f}s | Gen: {gen_time:.1f}s | Out: {out[:80]}..."


def test_bge_m3():
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    emb = model.encode(["What is photosynthesis?"])
    enc_time = time.perf_counter() - t0
    dim = emb.shape[1]
    del model
    return f"Load: {load_time:.1f}s | Encode: {enc_time:.2f}s | Dim: {dim}"


def test_reranker():
    from sentence_transformers import CrossEncoder

    t0 = time.perf_counter()
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = model.predict(
        [("What is photosynthesis?", "Plants use sunlight to make food.")]
    )
    rank_time = time.perf_counter() - t0
    del model
    return f"Load: {load_time:.1f}s | Rank: {rank_time:.3f}s | Score: {scores[0]:.4f}"


def test_indictrans2():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "ai4bharat/indictrans2-en-indic-1B"
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
    load_time = time.perf_counter() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    del model, tokenizer
    return f"Load: {load_time:.1f}s | Params: {param_count:.0f}M"


def test_whisper():
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_id = "openai/whisper-large-v3-turbo"
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    load_time = time.perf_counter() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    del model, processor
    return f"Load: {load_time:.1f}s | Params: {param_count:.0f}M"


def test_got_ocr():
    from transformers import AutoModel, AutoTokenizer

    model_id = "ucaslcl/GOT-OCR2_0"
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, low_cpu_mem_usage=True
    )
    load_time = time.perf_counter() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    del model, tokenizer
    return f"Load: {load_time:.1f}s | Params: {param_count:.0f}M"


def test_mms_tts():
    from transformers import VitsModel, AutoTokenizer

    model_id = "facebook/mms-tts-hin"
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    inputs = tokenizer("नमस्ते दुनिया", return_tensors="pt")
    import torch

    with torch.no_grad():
        output = model(**inputs)
    tts_time = time.perf_counter() - t0
    audio_len = output.waveform.shape[1] / model.config.sampling_rate
    del model, tokenizer
    return f"Load: {load_time:.1f}s | TTS: {tts_time:.2f}s | Audio: {audio_len:.2f}s"


if __name__ == "__main__":
    print("=" * 60)
    print("  Shiksha Setu — Full Model Verification")
    print("  7 Models • All Local • Apple Silicon")
    print("=" * 60)
    print()

    tests = [
        ("Qwen3-8B (MLX 4-bit) — LLM + Validation", test_qwen3),
        ("BGE-M3 — Embeddings (1024D)", test_bge_m3),
        ("BGE-Reranker-v2-M3 — Reranking", test_reranker),
        ("IndicTrans2-1B — Translation", test_indictrans2),
        ("Whisper Large V3 Turbo — STT", test_whisper),
        ("GOT-OCR2.0 — Document OCR", test_got_ocr),
        ("MMS-TTS (Hindi) — Text-to-Speech", test_mms_tts),
    ]

    for i, (name, fn) in enumerate(tests, 1):
        test_model(name, i, len(tests), fn)

    # Summary
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, status in results.items():
        print(f"  {status}  {name}")
        if "❌" in status:
            all_pass = False
    print()
    if all_pass:
        print("  ✅ ALL 7 MODELS VERIFIED — System ready!")
    else:
        failed = sum(1 for s in results.values() if "❌" in s)
        print(f"  ⚠️  {failed} model(s) failed verification")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)

#!/usr/bin/env python3
"""Quick test script to verify models are working."""

import sys

sys.path.insert(0, ".")


def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    try:
        import torch

        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    MPS available: {torch.backends.mps.is_available()}")

        import transformers

        print(f"  ✓ Transformers {transformers.__version__}")

        from sentence_transformers import SentenceTransformer

        print("  ✓ SentenceTransformers")

        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_embedding_model():
    """Test BGE-M3 embedding model."""
    print("\nTesting BGE-M3 embeddings...")
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3")

        # Test embedding
        text = "Photosynthesis is the process by which plants make food."
        embedding = model.encode(text)

        print(f"  ✓ Embedding dimension: {len(embedding)}")
        print(f"  ✓ Sample: {embedding[:5]}...")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_simplification_model():
    """Test Qwen3-8B simplification model via MLX."""
    print("\nTesting Qwen3-8B (simplification + validation)...")
    try:
        import mlx_lm

        model_id = "mlx-community/Qwen3-8B-4bit"

        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("  Loading model (this may take a minute)...")

        # Use MPS on Mac
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "mps":
            model = model.to(device)

        # Test generation
        prompt = (
            "Simplify this for a 5th grader: Photosynthesis is the biochemical process."
        )
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        print("  Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, do_sample=True, temperature=0.7
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  ✓ Generated: {response[-200:]}")

        # Cleanup
        del model
        torch.mps.empty_cache() if device == "mps" else None

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_translation_model():
    """Test IndicTrans2 translation model."""
    print("\nTesting IndicTrans2 (translation)...")
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = "ai4bharat/indictrans2-en-indic-1B"

        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        print("  Loading model...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )

        if device in ("mps", "cpu"):
            model = model.to(device)

        # Test translation (English -> Hindi)
        text = "Hello, how are you?"
        inputs = tokenizer(text, return_tensors="pt").to(device)

        print("  Translating...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  ✓ Translation: {translation}")

        # Cleanup
        del model
        torch.mps.empty_cache() if device == "mps" else None

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_validation_model():
    """Test validation via Qwen3-8B (shared with simplification)."""
    print("\nTesting Qwen3-8B (validation — shared with simplification)...")
    try:
        import mlx_lm

        model_id = "mlx-community/Qwen3-8B-4bit"

        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("  Loading model...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "mps":
            model = model.to(device)

        # Test validation prompt
        prompt = (
            "Is this text appropriate for complexity level 5? 'Plants make food using sunlight.'"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print("  Generating...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  ✓ Response: {response[-150:]}")

        # Cleanup
        del model
        torch.mps.empty_cache() if device == "mps" else None

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_database():
    """Test database connection (absorbed from test_setup.py)."""
    print("\nTesting database connection...")
    try:
        import os

        from dotenv import load_dotenv
        from sqlalchemy import create_engine, text

        load_dotenv()
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/shiksha_setu",
        )

        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("  ✓ Database connection successful")
        return True
    except Exception as e:
        print(f"  ✗ Database test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ShikshaSetu Model & Environment Test")
    print("=" * 60)

    results = {}

    results["imports"] = test_imports()
    results["database"] = test_database()
    results["embeddings"] = test_embedding_model()

    # Only test heavy models if explicitly requested
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        results["simplification"] = test_simplification_model()
        results["translation"] = test_translation_model()
        results["validation"] = test_validation_model()
    else:
        print("\nSkipping heavy model tests. Run with --full to test all models.")

    print("\n" + "=" * 60)
    print("Results:")
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print("=" * 60)
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    sys.exit(0 if all_passed else 1)

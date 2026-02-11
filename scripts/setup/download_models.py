#!/usr/bin/env python3
"""Download and cache required ML models for the ShikshaSetu platform.

2025 Optimal Model Stack:
- Simplification + Validation: Qwen3-8B (MLX 4-bit)
- Translation: IndicTrans2 1B
- TTS: MMS-TTS (facebook/mms-tts-*)
- STT: Whisper Large V3 Turbo
- OCR: GOT-OCR2.0
- Embeddings: BGE-M3
- Reranker: BGE-Reranker-v2-M3
"""

import os
import sys
from pathlib import Path


def download_models():
    """Download required AI models for the ShikshaSetu platform."""
    print("=" * 60)
    print("ShikshaSetu Model Downloader - 2025 Optimal Stack")
    print("=" * 60)

    # Set cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n📁 Cache directory: {cache_dir}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\n✓ Transformers library loaded successfully")
    except ImportError:
        print("\n✗ Error: transformers library not installed")
        print("Run: pip install transformers sentencepiece")
        sys.exit(1)

    # Model configurations - 2025 Optimal Stack
    models = [
        {
            "name": "Qwen3-8B (MLX 4-bit)",
            "id": "mlx-community/Qwen3-8B-4bit",
            "description": "Text simplification, validation & grade-level adaptation",
            "size": "~4.6GB (MLX 4-bit)",
            "type": "causal_lm",
        },
        {
            "name": "IndicTrans2 1B",
            "id": "ai4bharat/indictrans2-en-indic-1B",
            "description": "10-language Indian translation",
            "size": "~2GB",
            "type": "translation",
        },

        {
            "name": "BGE-M3",
            "id": "BAAI/bge-m3",
            "description": "RAG embeddings (1024D, multilingual)",
            "size": "~1.2GB",
            "type": "embeddings",
        },
        {
            "name": "BGE-Reranker-v2-M3",
            "id": "BAAI/bge-reranker-v2-m3",
            "description": "Improve retrieval accuracy",
            "size": "~1GB",
            "type": "reranker",
        },
        {
            "name": "GOT-OCR2.0",
            "id": "ucaslcl/GOT-OCR2_0",
            "description": "High-accuracy text extraction from images",
            "size": "~1.5GB",
            "type": "ocr",
        },
    ]

    print("\n" + "=" * 60)
    print("2025 Optimal Model Stack:")
    print("=" * 60)
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   ID: {model['id']}")
        print(f"   Purpose: {model['description']}")
        print(f"   Size: {model['size']}")

    print("\n" + "=" * 60)
    print("\nNote: MMS-TTS models are downloaded automatically on first use.")
    print("      STT uses openai/whisper-large-v3-turbo.")
    print("=" * 60)

    choice = (
        input("\nDownload which models? (1-6, 'all', or 'essential'): ").strip().lower()
    )

    models_to_download = []
    if choice == "all":
        models_to_download = models
    elif choice == "essential":
        # Essential: Simplification + Embeddings + Translation
        models_to_download = [models[0], models[1], models[3]]
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            models_to_download = [models[i] for i in indices if 0 <= i < len(models)]
        except (ValueError, IndexError):
            print("Invalid choice. Downloading essential models.")
            models_to_download = [models[0], models[1], models[3]]

    print("\n" + "=" * 60)
    print("Downloading selected models...")
    print("=" * 60)

    for model in models_to_download:
        print(f"\n🔄 Downloading {model['name']}...")
        print(f"   Model ID: {model['id']}")

        try:
            # Download tokenizer
            print("   → Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(
                model["id"], cache_dir=cache_dir, trust_remote_code=True
            )
            print("   ✓ Tokenizer downloaded")

            # Download model based on type
            print("   → Loading model (this may take a while)...")
            if model["type"] == "causal_lm":
                _model_obj = AutoModelForCausalLM.from_pretrained(
                    model["id"],
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto",
                )
            elif model["type"] == "embeddings":
                from sentence_transformers import SentenceTransformer

                _model_obj = SentenceTransformer(
                    model["id"], cache_folder=str(cache_dir)
                )
            elif model["type"] == "reranker":
                from sentence_transformers import CrossEncoder

                _model_obj = CrossEncoder(model["id"])
            else:
                # Generic loading
                from transformers import AutoModel

                _model_obj = AutoModel.from_pretrained(
                    model["id"], cache_dir=cache_dir, trust_remote_code=True
                )

            print("   ✓ Model downloaded successfully")

        except Exception as e:
            print(f"   ⚠️  Error downloading {model['name']}: {e}")
            print("   You can manually download later with:")
            print(f"   huggingface-cli download {model['id']}")
            continue

    print("\n" + "=" * 60)
    print("✓ Model download complete!")
    print("=" * 60)

    print("\n🚀 Ready to start the application!")
    print("\nNext steps:")
    print("1. Ensure DATABASE_URL is set in .env")
    print("2. Run migrations: alembic upgrade head")
    print("3. Start the server: ./start.sh")


if __name__ == "__main__":
    download_models()

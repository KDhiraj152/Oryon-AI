#!/usr/bin/env python3
"""Download and cache required ML models for the ShikshaSetu platform."""
import os
import sys
from pathlib import Path

def download_models():
    """Download FLAN-T5 and other required models."""
    print("=" * 60)
    print("ShikshaSetu Model Downloader")
    print("=" * 60)
    
    # Set cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\n📁 Cache directory: {cache_dir}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("\n✓ Transformers library loaded successfully")
    except ImportError:
        print("\n✗ Error: transformers library not installed")
        print("Run: pip install transformers sentencepiece")
        sys.exit(1)
    
    # Model configurations
    models = [
        {
            "name": "FLAN-T5 Base",
            "id": "google/flan-t5-base",
            "description": "Text simplification and adaptation",
            "size": "~900MB"
        },
        {
            "name": "FLAN-T5 Small",
            "id": "google/flan-t5-small",
            "description": "Lightweight alternative for testing",
            "size": "~300MB"
        }
    ]
    
    print("\n" + "=" * 60)
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   ID: {model['id']}")
        print(f"   Purpose: {model['description']}")
        print(f"   Size: {model['size']}")
    
    print("\n" + "=" * 60)
    choice = input("\nWhich model to download? (1/2/both) [both]: ").strip().lower()
    
    if not choice:
        choice = "both"
    
    models_to_download = []
    if choice in ["1", "both"]:
        models_to_download.append(models[0])
    if choice in ["2", "both"]:
        models_to_download.append(models[1])
    
    if not models_to_download:
        print("Invalid choice. Defaulting to both models.")
        models_to_download = models
    
    print("\n" + "=" * 60)
    print("Downloading models...")
    print("=" * 60)
    
    for model in models_to_download:
        print(f"\n🔄 Downloading {model['name']}...")
        print(f"   Model ID: {model['id']}")
        
        try:
            # Download tokenizer
            print(f"   → Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model['id'],
                cache_dir=cache_dir
            )
            print(f"   ✓ Tokenizer downloaded")
            
            # Download model
            print(f"   → Loading model (this may take a while)...")
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(
                model['id'],
                cache_dir=cache_dir
            )
            print(f"   ✓ Model downloaded successfully")
            
            # Test the model
            print(f"   → Testing model...")
            test_input = "Simplify this text for grade 5 students: Photosynthesis is the process by which plants convert light energy into chemical energy."
            inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
            outputs = model_obj.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   ✓ Model test passed")
            print(f"   Test output: {result[:100]}...")
            
        except Exception as e:
            print(f"   ✗ Error downloading {model['name']}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✓ Model download complete!")
    print("=" * 60)
    
    # Create .env file if it doesn't exist
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("\n📝 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# ShikshaSetu Environment Variables\n\n")
            f.write("# Hugging Face API Configuration\n")
            f.write("HUGGINGFACE_API_KEY=your_api_key_here\n\n")
            f.write("# Model IDs\n")
            f.write("FLANT5_MODEL_ID=google/flan-t5-base\n")
            f.write("INDICTRANS2_MODEL_ID=ai4bharat/indictrans2-en-indic-1B\n")
            f.write("BERT_MODEL_ID=bert-base-multilingual-cased\n")
            f.write("VITS_MODEL_ID=facebook/mms-tts-hin\n\n")
            f.write("# Database Configuration\n")
            f.write("DATABASE_URL=postgresql://postgres:password@localhost:5432/education_content\n\n")
            f.write("# API Configuration\n")
            f.write("FLASK_PORT=5000\n")
            f.write("FASTAPI_PORT=8000\n")
        print(f"   ✓ Created {env_file}")
        print("   ⚠️  Remember to add your HUGGINGFACE_API_KEY to .env file")
    
    print("\n🚀 Ready to start the application!")
    print("\nNext steps:")
    print("1. Add your Hugging Face API key to .env file")
    print("2. Set up the database: alembic upgrade head")
    print("3. Start the FastAPI server: uvicorn src.api.fastapi_app:app --reload")
    print("4. Or start Flask server: python -m src.api.flask_app")

if __name__ == "__main__":
    download_models()

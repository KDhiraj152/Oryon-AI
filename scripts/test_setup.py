#!/usr/bin/env python3
"""Quick test to verify all dependencies and FLAN-T5 model are working."""
import sys

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    try:
        import flask
        print("✓ Flask")
        import fastapi
        print("✓ FastAPI")
        import sqlalchemy
        print("✓ SQLAlchemy")
        import transformers
        print("✓ Transformers")
        import torch
        print("✓ PyTorch")
        import sentencepiece
        print("✓ SentencePiece")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_flan_t5():
    """Test FLAN-T5 model."""
    print("\nTesting FLAN-T5 model...")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_id = "google/flan-t5-base"
        print(f"Loading model: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        test_text = "Simplify this for grade 5: The mitochondria is the powerhouse of the cell."
        inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=100)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✓ Model loaded and tested successfully")
        print(f"Input: {test_text}")
        print(f"Output: {result}")
        return True
    except Exception as e:
        print(f"✗ FLAN-T5 test failed: {e}")
        return False

def test_database():
    """Test database connection."""
    print("\nTesting database connection...")
    try:
        import os
        from dotenv import load_dotenv
        from sqlalchemy import create_engine, text
        
        load_dotenv()
        db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/education_content')
        
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ShikshaSetu Dependency Test")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "FLAN-T5 Model": test_flan_t5(),
        "Database": test_database()
    }
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Run: ./start.sh")
        print("2. Or manually: uvicorn src.api.fastapi_app:app --reload")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

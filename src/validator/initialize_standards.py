#!/usr/bin/env python3
"""Utility script to initialize NCERT standards database."""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .ncert_standards import initialize_ncert_standards
from ..database import get_db


def main():
    """Initialize NCERT standards database."""
    logger.info("Initializing NCERT standards database...")
    
    try:
        # Ensure database tables exist
        db = get_db()
        db.create_tables()
        logger.info("Database tables created/verified")
        
        # Initialize standards
        loader = initialize_ncert_standards()
        logger.info(f"Successfully initialized {len(loader.standards)} NCERT standards")
        
        # Test the loader
        test_content = "Understanding fractions and their operations in mathematics"
        matches = loader.find_matching_standards(test_content, 6, "Mathematics", top_k=3)
        
        logger.info(f"\nTest query results ({len(matches)} matches):")
        for standard, score in matches:
            logger.info(f"  - {standard.topic} (Grade {standard.grade_level}): {score:.3f}")
        
        logger.info("\nNCERT standards database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing NCERT standards: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
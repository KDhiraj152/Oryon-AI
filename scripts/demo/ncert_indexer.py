#!/usr/bin/env python3
"""
NCERT Textbook Pre-indexing Script

Batch processes NCERT textbooks to:
1. Extract text from PDFs
2. Generate embeddings
3. Store in pgvector database
4. Create searchable index

Usage:
    python scripts/ncert_indexer.py --curriculum data/curriculum/ --batch-size 100
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import uuid
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import get_db_session
from backend.models import ProcessedContent, Document, Embedding
from backend.services.rag import RAGService
from backend.utils.logging import get_logger
import PyPDF2

logger = get_logger(__name__)


class NCERTIndexer:
    """Index NCERT textbooks for RAG system."""
    
    def __init__(self, curriculum_dir: Path, batch_size: int = 100):
        self.curriculum_dir = curriculum_dir
        self.batch_size = batch_size
        self.rag_service = RAGService()
        self.stats = {
            "total_books": 0,
            "total_pages": 0,
            "total_embeddings": 0,
            "failed_books": 0
        }
    
    def index_curriculum(self):
        """Index all NCERT textbooks in curriculum directory."""
        logger.info(f"Starting NCERT indexing from: {self.curriculum_dir}")
        
        # Find all curriculum JSON files
        json_files = list(self.curriculum_dir.glob("*.json"))
        
        if not json_files:
            logger.warning("No curriculum JSON files found")
            return
        
        for json_file in json_files:
            try:
                self._index_curriculum_file(json_file)
            except Exception as e:
                logger.error(f"Failed to index {json_file}: {e}")
                self.stats["failed_books"] += 1
        
        # Print summary
        logger.info("=" * 60)
        logger.info("NCERT Indexing Complete")
        logger.info(f"Total books processed: {self.stats['total_books']}")
        logger.info(f"Total pages indexed: {self.stats['total_pages']}")
        logger.info(f"Total embeddings created: {self.stats['total_embeddings']}")
        logger.info(f"Failed books: {self.stats['failed_books']}")
        logger.info("=" * 60)
    
    def _index_curriculum_file(self, json_file: Path):
        """Index a single curriculum JSON file."""
        logger.info(f"Processing: {json_file.name}")
        
        # Load curriculum metadata
        with open(json_file, 'r', encoding='utf-8') as f:
            curriculum = json.load(f)
        
        # Extract metadata
        grade = curriculum.get("grade", "unknown")
        subject = curriculum.get("subject", "unknown")
        books = curriculum.get("books", [])
        
        for book in books:
            self._index_book(book, grade, subject)
    
    def _index_book(self, book: Dict[str, Any], grade: str, subject: str):
        """Index a single textbook."""
        book_title = book.get("title", "Unknown")
        pdf_path = book.get("pdf_path")
        
        if not pdf_path or not Path(pdf_path).exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return
        
        logger.info(f"Indexing book: {book_title}")
        
        try:
            # Extract text from PDF
            text_chunks = self._extract_pdf_text(pdf_path)
            
            if not text_chunks:
                logger.warning(f"No text extracted from {book_title}")
                return
            
            # Create document record
            with get_db_session() as session:
                document = Document(
                    id=uuid.uuid4(),
                    title=book_title,
                    content="",  # Full content not stored, only embeddings
                    metadata={
                        "grade": grade,
                        "subject": subject,
                        "pdf_path": pdf_path,
                        "chapter_count": len(text_chunks),
                        "indexed_at": datetime.now(timezone.utc).isoformat()
                    }
                )
                session.add(document)
                session.commit()
                
                document_id = str(document.id)
            
            # Generate embeddings in batches
            embeddings_created = 0
            for i in range(0, len(text_chunks), self.batch_size):
                batch = text_chunks[i:i + self.batch_size]
                
                # Store embeddings
                for idx, chunk in enumerate(batch):
                    chunk_index = i + idx
                    self.rag_service.add_document_chunk(
                        document_id=document_id,
                        text=chunk,
                        chunk_index=chunk_index,
                        metadata={
                            "page": chunk_index,
                            "book_title": book_title
                        }
                    )
                    embeddings_created += 1
                
                logger.info(f"Progress: {i + len(batch)}/{len(text_chunks)} chunks")
            
            # Update statistics
            self.stats["total_books"] += 1
            self.stats["total_pages"] += len(text_chunks)
            self.stats["total_embeddings"] += embeddings_created
            
            logger.info(f"✓ Indexed {book_title}: {embeddings_created} embeddings")
        
        except Exception as e:
            logger.error(f"Failed to index {book_title}: {e}", exc_info=True)
            self.stats["failed_books"] += 1
    
    def _extract_pdf_text(self, pdf_path: str) -> List[str]:
        """Extract text from PDF, split into chunks."""
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text.strip():
                        # Split into paragraphs (simple chunking)
                        paragraphs = text.split('\n\n')
                        for para in paragraphs:
                            if len(para.strip()) > 50:  # Min 50 chars
                                chunks.append(para.strip())
        
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
        
        return chunks


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Index NCERT textbooks for RAG")
    parser.add_argument(
        "--curriculum",
        type=str,
        default="data/curriculum",
        help="Path to curriculum directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation"
    )
    
    args = parser.parse_args()
    
    curriculum_dir = Path(args.curriculum)
    if not curriculum_dir.exists():
        logger.error(f"Curriculum directory not found: {curriculum_dir}")
        sys.exit(1)
    
    indexer = NCERTIndexer(curriculum_dir, args.batch_size)
    indexer.index_curriculum()


if __name__ == "__main__":
    main()

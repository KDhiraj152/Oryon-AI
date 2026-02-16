#!/usr/bin/env python3
"""
Document Pre-indexing Script

Batch processes documents to:
1. Extract text from PDFs
2. Generate embeddings
3. Store in pgvector database
4. Create searchable index

Usage:
    python scripts/content_domain_indexer.py --content_domain data/content_domain/ --batch-size 100
"""
import argparse
import json
import sys
import uuid
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import PyPDF2

from backend.database import get_db_session
from backend.models import DocumentChunk, Embedding, ProcessedContent

# Alias for backward compatibility - Document model was renamed to DocumentChunk
Document = DocumentChunk
from backend.services.rag import RAGService
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentIndexer:
    """Index documents for RAG system."""
    
    def __init__(self, content_domain_dir: Path, batch_size: int = 100):
        self.content_domain_dir = content_domain_dir
        self.batch_size = batch_size
        self.rag_service = RAGService()
        self.stats = {
            "total_books": 0,
            "total_pages": 0,
            "total_embeddings": 0,
            "failed_books": 0
        }
    
    def index_content_domain(self):
        """Index all documents in content directory."""
        logger.info(f"Starting document indexing from: {self.content_domain_dir}")
        
        # Find all content JSON files
        json_files = list(self.content_domain_dir.glob("*.json"))
        
        if not json_files:
            logger.warning("No content JSON files found")
            return
        
        for json_file in json_files:
            try:
                self._index_content_domain_file(json_file)
            except Exception as e:
                logger.error(f"Failed to index {json_file}: {e}")
                self.stats["failed_books"] += 1
        
        # Print summary
        logger.info("=" * 60)
        logger.info("content_domain Indexing Complete")
        logger.info(f"Total books processed: {self.stats['total_books']}")
        logger.info(f"Total pages indexed: {self.stats['total_pages']}")
        logger.info(f"Total embeddings created: {self.stats['total_embeddings']}")
        logger.info(f"Failed books: {self.stats['failed_books']}")
        logger.info("=" * 60)
    
    def _index_content_domain_file(self, json_file: Path):
        """Index a single content JSON file."""
        logger.info(f"Processing: {json_file.name}")
        
        # Load content metadata
        with open(json_file, encoding='utf-8') as f:
            content_domain = json.load(f)
        
        # Extract metadata
        grade = content_domain.get("grade", "unknown")
        subject = content_domain.get("subject", "unknown")
        books = content_domain.get("books", [])
        
        for book in books:
            self._index_book(book, grade, subject)
    
    def _index_book(self, book: dict[str, Any], grade: str, subject: str):
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
                        "indexed_at": datetime.now(UTC).isoformat()
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
    
    def _extract_pdf_text(self, pdf_path: str) -> list[str]:
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
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument(
        "--content_domain",
        type=str,
        default="data/content_domain",
        help="Path to content directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation"
    )
    
    args = parser.parse_args()
    
    content_domain_dir = Path(args.content_domain)
    if not content_domain_dir.exists():
        logger.error(f"Curriculum directory not found: {content_domain_dir}")
        sys.exit(1)
    
    indexer = DocumentIndexer(content_domain_dir, args.batch_size)
    indexer.index_content_domain()


if __name__ == "__main__":
    main()

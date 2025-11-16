"""
RAG (Retrieval-Augmented Generation) Service for Q&A functionality.

This service handles:
- Text chunking from documents
- Embedding generation using sentence-transformers
- Vector similarity search using pgvector
- Context retrieval for question answering
"""

import logging
from typing import List, Dict, Tuple, Optional
from uuid import UUID
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from sentence_transformers import SentenceTransformer

from ..database import SessionLocal, get_db
from ..models import DocumentChunk, Embedding, ProcessedContent

logger = logging.getLogger(__name__)


class RAGService:
    """Service for Retrieval-Augmented Generation operations."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG service with embedding model.
        
        Args:
            embedding_model: HuggingFace model name for embeddings
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 default
        
    def _load_embedding_model(self):
        """Lazy load the embedding model."""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 512, 
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct != -1:
                        end = last_punct + len(punct)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < text_length else text_length
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        self._load_embedding_model()
        
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def store_document_chunks(
        self,
        content_id: UUID,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Chunk document and store chunks with embeddings in database.
        
        Args:
            content_id: ID of the processed content
            text: Full document text
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            metadata: Optional metadata for chunks (page numbers, etc.)
            
        Returns:
            Number of chunks created
        """
        logger.info(f"Storing chunks for content {content_id}")
        
        # Chunk the text
        chunks = self.chunk_text(text, chunk_size, overlap)
        
        if not chunks:
            logger.warning(f"No chunks generated for content {content_id}")
            return 0
        
        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = [self.generate_embedding(chunk) for chunk in chunks]
        
        # Store in database
        session = SessionLocal()
        try:
            for idx, (chunk_text, embedding_vector) in enumerate(zip(chunks, embeddings)):
                # Create chunk record
                chunk = DocumentChunk(
                    content_id=content_id,
                    chunk_index=idx,
                    chunk_text=chunk_text,
                    chunk_size=len(chunk_text),
                    chunk_metadata=metadata or {}
                )
                session.add(chunk)
                session.flush()  # Get chunk ID
                
                # Create embedding record with vector
                embedding_record = Embedding(
                    chunk_id=chunk.id,
                    content_id=content_id,
                    embedding_model=self.embedding_model_name
                )
                session.add(embedding_record)
                session.flush()
                
                # Update vector column using raw SQL
                session.execute(
                    text("UPDATE embeddings SET embedding = :vector::vector WHERE id = :id"),
                    {"vector": str(embedding_vector), "id": str(embedding_record.id)}
                )
            
            session.commit()
            logger.info(f"Stored {len(chunks)} chunks with embeddings for content {content_id}")
            return len(chunks)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing chunks: {e}")
            raise
        finally:
            session.close()
    
    def search_similar_chunks(
        self,
        query: str,
        content_id: UUID,
        top_k: int = 3,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for chunks similar to query using vector similarity.
        
        Args:
            query: Question or search query
            content_id: ID of content to search within
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        logger.info(f"Searching for chunks similar to: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search using pgvector cosine similarity
        session = SessionLocal()
        try:
            # Using raw SQL for vector similarity search
            sql = text("""
                SELECT 
                    dc.id, 
                    dc.content_id,
                    dc.chunk_index,
                    dc.chunk_text,
                    dc.chunk_size,
                    dc.chunk_metadata,
                    dc.created_at,
                    1 - (e.embedding <=> :query_vector::vector) as similarity
                FROM document_chunks dc
                JOIN embeddings e ON dc.id = e.chunk_id
                WHERE dc.content_id = :content_id
                  AND 1 - (e.embedding <=> :query_vector::vector) >= :threshold
                ORDER BY e.embedding <=> :query_vector::vector
                LIMIT :limit
            """)
            
            result = session.execute(
                sql,
                {
                    "query_vector": str(query_embedding),
                    "content_id": str(content_id),
                    "threshold": similarity_threshold,
                    "limit": top_k
                }
            )
            
            results = []
            for row in result:
                chunk = DocumentChunk(
                    id=row.id,
                    content_id=row.content_id,
                    chunk_index=row.chunk_index,
                    chunk_text=row.chunk_text,
                    chunk_size=row.chunk_size,
                    chunk_metadata=row.chunk_metadata,
                    created_at=row.created_at
                )
                similarity = float(row.similarity)
                results.append((chunk, similarity))
            
            logger.info(f"Found {len(results)} similar chunks (similarity >= {similarity_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            raise
        finally:
            session.close()
    
    def retrieve_context(
        self,
        question: str,
        content_id: UUID,
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        Retrieve relevant context for answering a question.
        
        Args:
            question: User's question
            content_id: ID of content to search
            top_k: Number of chunks to retrieve
            
        Returns:
            Dict with context_text, chunk_ids, and scores
        """
        # Search for similar chunks
        similar_chunks = self.search_similar_chunks(question, content_id, top_k)
        
        if not similar_chunks:
            return {
                "context_text": "",
                "chunk_ids": [],
                "scores": [],
                "has_context": False
            }
        
        # Combine chunks into context
        context_parts = []
        chunk_ids = []
        scores = []
        
        for chunk, score in similar_chunks:
            context_parts.append(chunk.chunk_text)
            chunk_ids.append(chunk.id)
            scores.append(score)
        
        context_text = "\n\n".join(context_parts)
        
        return {
            "context_text": context_text,
            "chunk_ids": chunk_ids,
            "scores": scores,
            "has_context": True,
            "avg_score": sum(scores) / len(scores) if scores else 0.0
        }
    
    def delete_document_chunks(self, content_id: UUID) -> int:
        """
        Delete all chunks and embeddings for a document.
        
        Args:
            content_id: ID of content to delete chunks for
            
        Returns:
            Number of chunks deleted
        """
        session = SessionLocal()
        try:
            # Delete embeddings (will cascade to chunks due to foreign key)
            result = session.query(Embedding).filter(
                Embedding.content_id == content_id
            ).delete()
            
            session.commit()
            logger.info(f"Deleted {result} chunks/embeddings for content {content_id}")
            return result
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting chunks: {e}")
            raise
        finally:
            session.close()


# Global RAG service instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

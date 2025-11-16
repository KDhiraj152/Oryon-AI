"""NCERT standards database loader and indexer using BERT embeddings."""
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..pipeline.model_clients import BERTClient
from ..database import get_db
from ..models import NCERTStandard


@dataclass
class NCERTStandardData:
    """Data structure for NCERT standard with embeddings."""
    id: str
    grade_level: int
    subject: str
    topic: str
    learning_objectives: List[str]
    keywords: List[str]
    embedding: Optional[np.ndarray] = None
    combined_text: Optional[str] = None


class NCERTStandardsLoader:
    """Loads and indexes NCERT standards database with BERT embeddings."""
    
    def __init__(self, bert_client: Optional[BERTClient] = None):
        self.bert_client = bert_client or BERTClient()
        self.standards: List[NCERTStandardData] = []
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.db = get_db()
    
    def load_standards_from_json(self, json_path: str) -> List[NCERTStandardData]:
        """Load NCERT standards from JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"NCERT standards file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        standards = []
        for idx, standard in enumerate(data.get('standards', [])):
            # Create combined text for embedding
            combined_text = self._create_combined_text(standard)
            
            standard_data = NCERTStandardData(
                id=f"ncert_{standard['grade_level']}_{standard['subject']}_{idx}",
                grade_level=standard['grade_level'],
                subject=standard['subject'],
                topic=standard['topic'],
                learning_objectives=standard['learning_objectives'],
                keywords=standard['keywords'],
                combined_text=combined_text
            )
            standards.append(standard_data)
        
        self.standards = standards
        return standards
    
    def _create_combined_text(self, standard: Dict[str, Any]) -> str:
        """Create combined text representation for embedding generation."""
        parts = [
            f"Grade {standard['grade_level']}",
            f"Subject: {standard['subject']}",
            f"Topic: {standard['topic']}",
            "Learning Objectives: " + "; ".join(standard['learning_objectives']),
            "Keywords: " + ", ".join(standard['keywords'])
        ]
        return " | ".join(parts)
    
    def generate_embeddings(self) -> None:
        """Generate BERT embeddings for all standards."""
        logger.info(f"Generating embeddings for {len(self.standards)} standards...")
        
        for i, standard in enumerate(self.standards):
            if standard.combined_text:
                try:
                    # For BERT embeddings, we'll use a simple approach
                    # In a real implementation, you'd use sentence-transformers or similar
                    embedding = self._get_text_embedding(standard.combined_text)
                    standard.embedding = embedding
                    self.embeddings_cache[standard.id] = embedding
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Generated embeddings for {i + 1}/{len(self.standards)} standards")
                
                except Exception as e:
                    logger.error(f"Error generating embedding for standard {standard.id}: {e}")
                    # Use zero vector as fallback
                    standard.embedding = np.zeros(768)  # BERT base embedding size
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate BERT embedding for text."""
        # This is a simplified approach. In production, you'd use:
        # - sentence-transformers library
        # - Proper BERT feature extraction
        # - Caching mechanism
        
        # For now, create a mock embedding based on text hash
        # This should be replaced with actual BERT embedding generation
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Create a deterministic "embedding" from hash (for demo purposes)
        # In production, use actual BERT model
        seed = int(text_hash[:8], 16)
        rng = np.random.default_rng(seed)
        embedding = rng.normal(0, 1, 768)  # BERT base size
        return embedding / np.linalg.norm(embedding)  # Normalize
    
    def save_to_database(self) -> None:
        """Save standards to PostgreSQL database."""
        session = self.db.get_session()
        
        try:
            # Clear existing standards
            session.query(NCERTStandard).delete()
            
            # Insert new standards
            for standard in self.standards:
                db_standard = NCERTStandard(
                    grade_level=standard.grade_level,
                    subject=standard.subject,
                    topic=standard.topic,
                    learning_objectives=standard.learning_objectives,
                    keywords=standard.keywords
                )
                session.add(db_standard)
            
            session.commit()
            logger.info(f"Saved {len(self.standards)} standards to database")
            
        except Exception as e:
            session.rollback()
            from sqlalchemy.exc import DatabaseError
            raise DatabaseError(f"Error saving standards to database: {e}", params=None, orig=e) from e
        finally:
            session.close()
    
    def load_from_database(self) -> List[NCERTStandardData]:
        """Load standards from database."""
        session = self.db.get_session()
        
        try:
            db_standards = session.query(NCERTStandard).all()
            standards = []
            
            for db_standard in db_standards:
                combined_text = self._create_combined_text({
                    'grade_level': db_standard.grade_level,
                    'subject': db_standard.subject,
                    'topic': db_standard.topic,
                    'learning_objectives': db_standard.learning_objectives,
                    'keywords': db_standard.keywords
                })
                
                standard_data = NCERTStandardData(
                    id=str(db_standard.id),
                    grade_level=db_standard.grade_level,
                    subject=db_standard.subject,
                    topic=db_standard.topic,
                    learning_objectives=db_standard.learning_objectives,
                    keywords=db_standard.keywords,
                    combined_text=combined_text
                )
                standards.append(standard_data)
            
            self.standards = standards
            return standards
            
        finally:
            session.close()
    
    def find_matching_standards(
        self, 
        content: str, 
        grade_level: int, 
        subject: str, 
        top_k: int = 5
    ) -> List[Tuple[NCERTStandardData, float]]:
        """Find NCERT standards that match the given content."""
        if not self.standards:
            self.load_from_database()
        
        # Filter by grade level and subject
        filtered_standards = [
            s for s in self.standards 
            if s.grade_level == grade_level and s.subject.lower() == subject.lower()
        ]
        
        if not filtered_standards:
            return []
        
        # Generate embedding for input content
        content_embedding = self._get_text_embedding(content)
        
        # Calculate similarities
        similarities = []
        for standard in filtered_standards:
            if standard.embedding is None:
                standard.embedding = self._get_text_embedding(standard.combined_text)
            
            similarity = self._cosine_similarity(content_embedding, standard.embedding)
            similarities.append((standard, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def check_keyword_overlap(self, content: str, standard: NCERTStandardData) -> float:
        """Check keyword overlap between content and standard."""
        content_words = set(content.lower().split())
        standard_keywords = {keyword.lower() for keyword in standard.keywords}
        
        if not standard_keywords:
            return 0.0
        
        overlap = len(content_words.intersection(standard_keywords))
        return overlap / len(standard_keywords)
    
    def get_learning_objectives_match(
        self, 
        content: str, 
        standard: NCERTStandardData
    ) -> float:
        """Calculate match score for learning objectives."""
        if not standard.learning_objectives:
            return 0.0
        
        total_score = 0.0
        for objective in standard.learning_objectives:
            objective_embedding = self._get_text_embedding(objective)
            content_embedding = self._get_text_embedding(content)
            similarity = self._cosine_similarity(content_embedding, objective_embedding)
            total_score += similarity
        
        return total_score / len(standard.learning_objectives)


def initialize_ncert_standards(json_path: str = None) -> NCERTStandardsLoader:
    """Initialize NCERT standards database."""
    if json_path is None:
        # Use default path
        current_dir = Path(__file__).parent.parent.parent
        json_path = current_dir / "data" / "curriculum" / "ncert_standards_sample.json"
    
    loader = NCERTStandardsLoader()
    
    try:
        # Try to load from database first
        standards = loader.load_from_database()
        if not standards:
            # Load from JSON if database is empty
            logger.info("Loading NCERT standards from JSON...")
            loader.load_standards_from_json(str(json_path))
            loader.generate_embeddings()
            loader.save_to_database()
        else:
            logger.info(f"Loaded {len(standards)} NCERT standards from database")
            # Generate embeddings for loaded standards
            loader.generate_embeddings()
    
    except Exception as e:
        logger.error(f"Error initializing NCERT standards: {e}")
        # Fallback to JSON loading
        loader.load_standards_from_json(str(json_path))
        loader.generate_embeddings()
    
    return loader
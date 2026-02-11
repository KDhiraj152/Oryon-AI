"""content standards database loader and indexer using BGE-M3 embeddings."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ...database import get_db
from ...models import ContentStandard

if TYPE_CHECKING:
    from ..rag import BGEM3Embedder

logger = logging.getLogger(__name__)


@dataclass
class ContentStandardData:
    """Data structure for content_domain standard with embeddings."""

    id: str
    complexity_level: int
    subject: str
    topic: str
    learning_objectives: list[str]
    keywords: list[str]
    embedding: np.ndarray | None = None
    combined_text: str | None = None


class ContentStandardsLoader:
    """Loads and indexes content standards database with BGE-M3 embeddings."""

    def __init__(self, embedder: Optional["BGEM3Embedder"] = None):
        self._embedder = embedder
        self.standards: list[ContentStandardData] = []
        self.embeddings_cache: dict[str, np.ndarray] = {}
        self.db = get_db()

    def _get_embedder(self):
        """Lazy-load embedder using shared singleton if not provided."""
        if self._embedder is None:
            from ..rag import get_embedder

            self._embedder = (
                get_embedder()
            )  # Use singleton instead of creating new instance
        return self._embedder

    def load_standards_from_json(self, json_path: str) -> list[ContentStandardData]:
        """Load content standards from JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"content standards file not found: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        standards = []
        for idx, standard in enumerate(data.get("standards", [])):
            # Create combined text for embedding
            combined_text = self._create_combined_text(standard)

            standard_data = ContentStandardData(
                id=f"content_domain_{standard['complexity_level']}_{standard['subject']}_{idx}",
                complexity_level=standard["complexity_level"],
                subject=standard["subject"],
                topic=standard["topic"],
                learning_objectives=standard["learning_objectives"],
                keywords=standard["keywords"],
                combined_text=combined_text,
            )
            standards.append(standard_data)

        self.standards = standards
        return standards

    def _create_combined_text(self, standard: dict[str, Any]) -> str:
        """Create combined text representation for embedding generation."""
        parts = [
            f"Grade {standard['complexity_level']}",
            f"Subject: {standard['subject']}",
            f"Topic: {standard['topic']}",
            "Learning Objectives: " + "; ".join(standard["learning_objectives"]),
            "Keywords: " + ", ".join(standard["keywords"]),
        ]
        return " | ".join(parts)

    def generate_embeddings(self) -> None:
        """Generate BGE-M3 embeddings for all standards."""
        embedder = self._get_embedder()
        logger.info(f"Generating embeddings for {len(self.standards)} standards...")

        # Batch encode all combined texts for efficiency
        texts = [s.combined_text for s in self.standards if s.combined_text]

        if texts:
            try:
                embeddings = embedder.encode(texts)

                text_idx = 0
                for standard in self.standards:
                    if standard.combined_text:
                        standard.embedding = embeddings[text_idx]
                        self.embeddings_cache[standard.id] = embeddings[text_idx]
                        text_idx += 1

                logger.info(f"Generated {len(texts)} embeddings with BGE-M3")

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Fallback to zero vectors
                for standard in self.standards:
                    if standard.embedding is None:
                        standard.embedding = np.zeros(1024)  # BGE-M3 size

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate BGE-M3 embedding for text."""
        embedder = self._get_embedder()
        return embedder.encode([text])[0]

    def save_to_database(self) -> None:
        """Save standards to PostgreSQL database."""
        session = self.db.get_session()

        try:
            # Clear existing standards
            session.query(ContentStandard).delete()

            # Insert new standards
            for standard in self.standards:
                db_standard = ContentStandard(
                    complexity_level=standard.complexity_level,
                    subject=standard.subject,
                    topic=standard.topic,
                    learning_objectives=standard.learning_objectives,
                    keywords=standard.keywords,
                )
                session.add(db_standard)

            session.commit()
            logger.info(f"Saved {len(self.standards)} standards to database")

        except Exception as e:
            session.rollback()
            from sqlalchemy.exc import DatabaseError

            raise DatabaseError(
                f"Error saving standards to database: {e}", params=None, orig=e
            ) from e
        finally:
            session.close()

    def load_from_database(self) -> list[ContentStandardData]:
        """Load standards from database."""
        session = self.db.get_session()

        try:
            db_standards = session.query(ContentStandard).all()
            standards = []

            for db_standard in db_standards:
                combined_text = self._create_combined_text(
                    {
                        "complexity_level": db_standard.complexity_level,
                        "subject": db_standard.subject,
                        "topic": db_standard.topic,
                        "learning_objectives": db_standard.learning_objectives,
                        "keywords": db_standard.keywords,
                    }
                )

                standard_data = ContentStandardData(
                    id=str(db_standard.id),
                    complexity_level=db_standard.complexity_level,
                    subject=db_standard.subject,
                    topic=db_standard.topic,
                    learning_objectives=db_standard.learning_objectives,
                    keywords=db_standard.keywords,
                    combined_text=combined_text,
                )
                standards.append(standard_data)

            self.standards = standards
            return standards

        finally:
            session.close()

    def find_matching_standards(
        self, content: str, complexity_level: int, subject: str, top_k: int = 5
    ) -> list[tuple[ContentStandardData, float]]:
        """Find content standards that match the given content."""
        if not self.standards:
            self.load_from_database()

        # Filter by complexity level and subject
        filtered_standards = [
            s
            for s in self.standards
            if s.complexity_level == complexity_level and s.subject.lower() == subject.lower()
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
        """Calculate cosine similarity between two vectors.

        Hardware optimized: Uses SIMD operations when available.
        """
        # SIMD-optimized path
        try:
            from backend.core.optimized.simd_ops import cosine_similarity_single

            return float(cosine_similarity_single(vec1, vec2))
        except ImportError:
            pass

        # Fallback to numpy
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def check_keyword_overlap(self, content: str, standard: ContentStandardData) -> float:
        """Check keyword overlap between content and standard."""
        content_words = set(content.lower().split())
        standard_keywords = {keyword.lower() for keyword in standard.keywords}

        if not standard_keywords:
            return 0.0

        overlap = len(content_words.intersection(standard_keywords))
        return overlap / len(standard_keywords)

    def get_learning_objectives_match(
        self, content: str, standard: ContentStandardData
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


def initialize_content_standards(json_path: str | None = None) -> ContentStandardsLoader:
    """Initialize content standards database."""
    if json_path is None:
        # Use default path
        current_dir = Path(__file__).parent.parent.parent
        json_path = current_dir / "data" / "content_domain" / "content_standards_sample.json"

    loader = ContentStandardsLoader()

    try:
        # Try to load from database first
        standards = loader.load_from_database()
        if not standards:
            # Load from JSON if database is empty
            logger.info("Loading content standards from JSON...")
            loader.load_standards_from_json(str(json_path))
            loader.generate_embeddings()
            loader.save_to_database()
        else:
            logger.info(f"Loaded {len(standards)} content standards from database")
            # Generate embeddings for loaded standards
            loader.generate_embeddings()

    except Exception as e:
        logger.error(f"Error initializing content standards: {e}")
        # Fallback to JSON loading
        loader.load_standards_from_json(str(json_path))
        loader.generate_embeddings()

    return loader

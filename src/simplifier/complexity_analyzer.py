"""Complexity analysis utilities for text simplification."""
import re
import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetrics:
    """Detailed complexity metrics for text analysis."""
    avg_sentence_length: float
    avg_word_length: float
    avg_syllables_per_word: float
    total_sentences: int
    total_words: int
    complexity_score: float
    readability_level: str


class ComplexityAnalyzer:
    """
    Analyzes text complexity using multiple readability metrics.
    """
    
    # Readability level thresholds
    READABILITY_LEVELS = {
        (0.0, 0.3): 'Very Easy',
        (0.3, 0.5): 'Easy',
        (0.5, 0.7): 'Moderate',
        (0.7, 0.85): 'Difficult',
        (0.85, 1.0): 'Very Difficult'
    }
    
    def __init__(self):
        """Initialize the complexity analyzer."""
        logger.debug("ComplexityAnalyzer initialized")
    
    def analyze(self, text: str) -> ComplexityMetrics:
        """
        Perform comprehensive complexity analysis on text.
        
        Args:
            text: Text to analyze
        
        Returns:
            ComplexityMetrics with detailed analysis
        """
        if not text or len(text.strip()) == 0:
            return ComplexityMetrics(
                avg_sentence_length=0.0,
                avg_word_length=0.0,
                avg_syllables_per_word=0.0,
                total_sentences=0,
                total_words=0,
                complexity_score=0.0,
                readability_level='Unknown'
            )
        
        # Extract sentences and words
        sentences = self._split_sentences(text)
        words = self._split_words(text)
        
        if not sentences or not words:
            return ComplexityMetrics(
                avg_sentence_length=0.0,
                avg_word_length=0.0,
                avg_syllables_per_word=0.0,
                total_sentences=len(sentences),
                total_words=len(words),
                complexity_score=0.0,
                readability_level='Unknown'
            )
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Calculate overall complexity score
        complexity_score = self._calculate_complexity_score(
            avg_sentence_length,
            avg_word_length,
            avg_syllables
        )
        
        # Determine readability level
        readability_level = self._get_readability_level(complexity_score)
        
        logger.debug(
            f"Complexity analysis: sentences={len(sentences)}, words={len(words)}, "
            f"complexity={complexity_score:.2f}, level={readability_level}"
        )
        
        return ComplexityMetrics(
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            avg_syllables_per_word=avg_syllables,
            total_sentences=len(sentences),
            total_words=len(words),
            complexity_score=complexity_score,
            readability_level=readability_level
        )
    
    def _calculate_complexity_score(
        self,
        avg_sentence_length: float,
        avg_word_length: float,
        avg_syllables: float
    ) -> float:
        """
        Calculate overall complexity score from metrics.
        
        Args:
            avg_sentence_length: Average words per sentence
            avg_word_length: Average characters per word
            avg_syllables: Average syllables per word
        
        Returns:
            Complexity score (0-1)
        """
        # Normalize each metric to 0-1 scale
        
        # Sentence length: 5-30 words (typical range)
        sentence_score = min(max((avg_sentence_length - 5) / 25, 0), 1)
        
        # Word length: 3-10 characters (typical range)
        word_score = min(max((avg_word_length - 3) / 7, 0), 1)
        
        # Syllables: 1-4 syllables per word (typical range)
        syllable_score = min(max((avg_syllables - 1) / 3, 0), 1)
        
        # Weighted average (sentence structure matters most)
        complexity_score = (
            0.4 * sentence_score +
            0.3 * word_score +
            0.3 * syllable_score
        )
        
        return complexity_score
    
    def _get_readability_level(self, complexity_score: float) -> str:
        """
        Get readability level description from complexity score.
        
        Args:
            complexity_score: Complexity score (0-1)
        
        Returns:
            Readability level description
        """
        for (min_score, max_score), level in self.READABILITY_LEVELS.items():
            if min_score <= complexity_score <= max_score:
                return level
        
        return 'Unknown'
    
    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_words(self, text: str) -> list[str]:
        """
        Split text into words.
        
        Args:
            text: Text to split
        
        Returns:
            List of words
        """
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (approximate).
        
        Args:
            word: Word to analyze
        
        Returns:
            Estimated syllable count
        """
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        
        if len(word) == 0:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Every word has at least one syllable
        if syllable_count == 0:
            syllable_count = 1
        
        return syllable_count
    
    def get_grade_level_recommendation(self, complexity_score: float) -> int:
        """
        Recommend appropriate grade level based on complexity score.
        
        Args:
            complexity_score: Complexity score (0-1)
        
        Returns:
            Recommended grade level (5-12)
        """
        # Map complexity score to grade level
        if complexity_score < 0.3:
            return 5
        elif complexity_score < 0.4:
            return 6
        elif complexity_score < 0.5:
            return 7
        elif complexity_score < 0.6:
            return 8
        elif complexity_score < 0.7:
            return 9
        elif complexity_score < 0.8:
            return 10
        elif complexity_score < 0.9:
            return 11
        else:
            return 12

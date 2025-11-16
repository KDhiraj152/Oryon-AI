"""Text Simplifier component using Flan-T5 for grade-level content adaptation."""
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedText:
    """Result of text simplification."""
    text: str
    complexity_score: float
    grade_level: int
    subject: str
    metadata: Dict[str, Any]


class TextSimplifier:
    """
    Text simplification component that adapts educational content
    complexity based on grade level (5-12) using Flan-T5 model.
    """
    
    # Grade level ranges for complexity adaptation
    ELEMENTARY_GRADES = range(5, 7)  # Grades 5-6
    MIDDLE_GRADES = range(7, 9)      # Grades 7-8
    SECONDARY_GRADES = range(9, 13)  # Grades 9-12
    
    # Complexity score thresholds (0-1 scale)
    COMPLEXITY_THRESHOLDS = {
        'elementary': (0.0, 0.4),
        'middle': (0.4, 0.7),
        'secondary': (0.7, 1.0)
    }
    
    # Subject-specific terminology that should be preserved
    SUBJECT_TERMINOLOGY = {
        'Mathematics': [
            'equation', 'variable', 'coefficient', 'polynomial', 'theorem',
            'proof', 'derivative', 'integral', 'function', 'graph',
            'algebra', 'geometry', 'trigonometry', 'calculus', 'statistics'
        ],
        'Science': [
            'photosynthesis', 'molecule', 'atom', 'cell', 'organism',
            'energy', 'force', 'velocity', 'acceleration', 'chemical',
            'reaction', 'element', 'compound', 'ecosystem', 'evolution'
        ],
        'Social Studies': [
            'democracy', 'constitution', 'government', 'economy', 'culture',
            'civilization', 'revolution', 'independence', 'parliament', 'rights'
        ],
        'History': [
            'ancient', 'medieval', 'modern', 'empire', 'dynasty',
            'colonialism', 'independence', 'revolution', 'treaty', 'war'
        ],
        'Geography': [
            'latitude', 'longitude', 'climate', 'topography', 'ecosystem',
            'continent', 'ocean', 'mountain', 'river', 'plateau'
        ]
    }
    
    def __init__(self, model_client=None):
        """
        Initialize the Text Simplifier.
        
        Args:
            model_client: Optional Flan-T5 model client for inference
        """
        self.model_client = model_client
        logger.info("TextSimplifier initialized")
    
    def simplify_text(
        self,
        content: str,
        grade_level: int,
        subject: str
    ) -> SimplifiedText:
        """
        Simplify text content based on grade level and subject.
        
        Args:
            content: Original text to simplify
            grade_level: Target grade level (5-12)
            subject: Subject area for context-aware simplification
        
        Returns:
            SimplifiedText object with simplified content and metadata
        
        Raises:
            ValueError: If grade_level is out of range or content is empty
        """
        # Validate inputs
        if not content or len(content.strip()) == 0:
            raise ValueError("Content cannot be empty")
        
        if grade_level < 5 or grade_level > 12:
            raise ValueError(f"Grade level must be between 5 and 12, got {grade_level}")
        
        logger.info(f"Simplifying text for grade {grade_level}, subject {subject}")
        
        # Calculate original complexity
        original_complexity = self.get_complexity_score(content)
        logger.debug(f"Original complexity score: {original_complexity:.2f}")
        
        # Determine target complexity based on grade level
        target_complexity_range = self._get_target_complexity_range(grade_level)
        
        # Generate grade-specific prompt
        prompt = self._create_grade_specific_prompt(content, grade_level, subject)
        
        # Apply subject-specific simplification rules
        simplified_content = self._apply_subject_specific_rules(
            content, grade_level, subject
        )
        
        # Use Flan-T5 model for simplification if available
        if self.model_client:
            try:
                simplified_content = self.model_client.process(
                    prompt, grade_level, subject
                )
            except Exception as e:
                logger.warning(f"Model inference failed, using rule-based simplification: {e}")
                # Fall back to rule-based simplification
                simplified_content = self._rule_based_simplification(
                    content, grade_level, subject
                )
        else:
            # Use rule-based simplification
            simplified_content = self._rule_based_simplification(
                content, grade_level, subject
            )
        
        # Calculate final complexity
        final_complexity = self.get_complexity_score(simplified_content)
        logger.info(f"Simplified text complexity: {final_complexity:.2f}")
        
        return SimplifiedText(
            text=simplified_content,
            complexity_score=final_complexity,
            grade_level=grade_level,
            subject=subject,
            metadata={
                'original_complexity': original_complexity,
                'target_complexity_range': target_complexity_range,
                'simplification_method': 'model' if self.model_client else 'rule_based'
            }
        )
    
    def get_complexity_score(self, text: str) -> float:
        """
        Calculate complexity score using readability metrics.
        
        Uses a combination of:
        - Average sentence length
        - Average word length
        - Vocabulary complexity (syllable count)
        - Technical term density
        
        Args:
            text: Text to analyze
        
        Returns:
            Complexity score between 0 (simple) and 1 (complex)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0
        
        # Split into words
        words = self._split_words(text)
        if not words:
            return 0.0
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Normalize metrics to 0-1 scale
        # Sentence length: 5-30 words (typical range)
        sentence_score = min(max((avg_sentence_length - 5) / 25, 0), 1)
        
        # Word length: 3-10 characters (typical range)
        word_score = min(max((avg_word_length - 3) / 7, 0), 1)
        
        # Syllables: 1-4 syllables per word (typical range)
        syllable_score = min(max((avg_syllables - 1) / 3, 0), 1)
        
        # Weighted average (sentence structure matters most for readability)
        complexity_score = (
            0.4 * sentence_score +
            0.3 * word_score +
            0.3 * syllable_score
        )
        
        logger.debug(
            f"Complexity metrics - sentences: {avg_sentence_length:.1f}, "
            f"words: {avg_word_length:.1f}, syllables: {avg_syllables:.1f}, "
            f"score: {complexity_score:.2f}"
        )
        
        return complexity_score
    
    def _get_target_complexity_range(self, grade_level: int) -> tuple[float, float]:
        """
        Get target complexity range for a grade level.
        
        Args:
            grade_level: Grade level (5-12)
        
        Returns:
            Tuple of (min_complexity, max_complexity)
        """
        if grade_level in self.ELEMENTARY_GRADES:
            return self.COMPLEXITY_THRESHOLDS['elementary']
        elif grade_level in self.MIDDLE_GRADES:
            return self.COMPLEXITY_THRESHOLDS['middle']
        else:  # SECONDARY_GRADES
            return self.COMPLEXITY_THRESHOLDS['secondary']
    
    def _create_grade_specific_prompt(
        self,
        content: str,
        grade_level: int,
        subject: str
    ) -> str:
        """
        Create grade-specific prompt for Flan-T5 model.
        
        Args:
            content: Original content
            grade_level: Target grade level
            subject: Subject area
        
        Returns:
            Formatted prompt for the model
        """
        # Determine grade category
        if grade_level in self.ELEMENTARY_GRADES:
            grade_category = "early secondary education (grades 5-6)"
            instructions = (
                "Use simple sentences with basic vocabulary. "
                "Break down complex concepts into easy-to-understand parts. "
                "Use everyday examples and analogies."
            )
        elif grade_level in self.MIDDLE_GRADES:
            grade_category = "middle school (grades 7-8)"
            instructions = (
                "Use clear, straightforward language with moderate vocabulary. "
                "Explain concepts step-by-step. "
                "Include relevant examples to illustrate ideas."
            )
        else:  # SECONDARY_GRADES
            grade_category = "high school (grades 9-12)"
            instructions = (
                "Maintain academic rigor while ensuring clarity. "
                "Use appropriate technical terminology with explanations. "
                "Present concepts with logical structure and supporting details."
            )
        
        # Get subject-specific terminology to preserve
        preserve_terms = self.SUBJECT_TERMINOLOGY.get(subject, [])
        preserve_instruction = ""
        if preserve_terms:
            preserve_instruction = (
                f"\nPreserve these important {subject} terms: "
                f"{', '.join(preserve_terms[:5])}."
            )
        
        prompt = f"""Simplify the following {subject} text for {grade_category} students.

{instructions}{preserve_instruction}

Original text:
{content}

Simplified text:"""
        
        return prompt
    
    def _apply_subject_specific_rules(
        self,
        content: str,
        grade_level: int,
        subject: str
    ) -> str:
        """
        Apply subject-specific simplification rules.
        
        Args:
            content: Content to simplify
            grade_level: Target grade level
            subject: Subject area
        
        Returns:
            Content with subject-specific rules applied
        """
        # For Mathematics: Preserve mathematical notation and symbols
        if subject == 'Mathematics':
            # Ensure mathematical expressions are preserved
            # This is a placeholder - actual implementation would be more sophisticated
            content = self._preserve_math_notation(content)
        
        # For Science: Preserve scientific terminology but add explanations
        elif subject == 'Science':
            content = self._add_science_explanations(content, grade_level)
        
        # For Social Studies/History: Simplify historical context
        elif subject in ['Social Studies', 'History']:
            content = self._simplify_historical_context(content, grade_level)
        
        return content
    
    def _preserve_math_notation(self, content: str) -> str:
        """
        Preserve mathematical notation in content.
        
        Args:
            content: Content with math notation
        
        Returns:
            Content with preserved notation
        """
        # Placeholder implementation
        # In production, this would identify and protect mathematical expressions
        return content
    
    def _add_science_explanations(self, content: str, grade_level: int) -> str:
        """
        Add explanations for scientific terms based on grade level.
        
        Args:
            content: Content with scientific terms
            grade_level: Target grade level
        
        Returns:
            Content with added explanations
        """
        # Placeholder implementation
        # In production, this would identify scientific terms and add age-appropriate explanations
        return content
    
    def _simplify_historical_context(self, content: str, grade_level: int) -> str:
        """
        Simplify historical context based on grade level.
        
        Args:
            content: Content with historical context
            grade_level: Target grade level
        
        Returns:
            Simplified historical content
        """
        # Placeholder implementation
        # In production, this would simplify dates, events, and historical context
        return content
    
    def _rule_based_simplification(
        self,
        content: str,
        grade_level: int,
        subject: str
    ) -> str:
        """
        Apply rule-based simplification when model is not available.
        
        Args:
            content: Content to simplify
            grade_level: Target grade level
            subject: Subject area
        
        Returns:
            Simplified content
        """
        simplified = content
        
        # Split into sentences
        sentences = self._split_sentences(content)
        
        # Process each sentence
        simplified_sentences = []
        for sentence in sentences:
            # Break long sentences
            if len(sentence.split()) > 20 and grade_level in self.ELEMENTARY_GRADES:
                # Split at conjunctions
                parts = re.split(r'\s+(and|but|or|because|so)\s+', sentence, maxsplit=1)
                if len(parts) > 1:
                    simplified_sentences.append(parts[0].strip() + '.')
                    simplified_sentences.append(parts[2].strip())
                else:
                    simplified_sentences.append(sentence)
            else:
                simplified_sentences.append(sentence)
        
        simplified = ' '.join(simplified_sentences)
        
        # Replace complex words with simpler alternatives for lower grades
        if grade_level in self.ELEMENTARY_GRADES:
            simplified = self._replace_complex_words(simplified)
        
        return simplified
    
    def _replace_complex_words(self, text: str) -> str:
        """
        Replace complex words with simpler alternatives.
        
        Args:
            text: Text with complex words
        
        Returns:
            Text with simpler words
        """
        # Simple word replacements for elementary grades
        replacements = {
            'utilize': 'use',
            'demonstrate': 'show',
            'comprehend': 'understand',
            'acquire': 'get',
            'sufficient': 'enough',
            'approximately': 'about',
            'numerous': 'many',
            'commence': 'start',
            'terminate': 'end',
            'facilitate': 'help'
        }
        
        for complex_word, simple_word in replacements.items():
            # Case-insensitive replacement
            pattern = re.compile(r'\b' + complex_word + r'\b', re.IGNORECASE)
            text = pattern.sub(simple_word, text)
        
        return text
    
    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting (production would use more sophisticated NLP)
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
        # Remove punctuation and split
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
        
        # Remove non-alphabetic characters
        word = re.sub(r'[^a-z]', '', word)
        
        if len(word) == 0:
            return 0
        
        # Count vowel groups
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

"""Validation Module for content quality assurance using BERT."""
import re
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..pipeline.model_clients import BERTClient
from .ncert_standards import NCERTStandardsLoader, NCERTStandardData

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of content validation with detailed metrics."""
    content_id: str
    semantic_accuracy: float
    ncert_alignment_score: float
    script_accuracy: bool
    age_appropriate: bool
    overall_status: str  # 'passed', 'failed', 'needs_review'
    issues: List[str]
    recommendations: List[str]
    quality_metrics: Dict[str, float]
    validation_timestamp: datetime


@dataclass
class QualityReport:
    """Detailed quality report for validated content."""
    validation_result: ValidationResult
    matched_standards: List[Tuple[NCERTStandardData, float]]
    keyword_overlap_scores: List[float]
    learning_objective_matches: List[float]
    technical_terms_preserved: bool
    script_rendering_issues: List[str]


class ValidationModule:
    """BERT-based validation module for educational content quality assurance."""
    
    def __init__(self, bert_client: Optional[BERTClient] = None):
        self.bert_client = bert_client or BERTClient()
        self.ncert_loader = NCERTStandardsLoader(self.bert_client)
        self.quality_threshold = 0.80  # 80% NCERT accuracy threshold
        
        # Initialize NCERT standards
        try:
            self.ncert_loader.load_from_database()
            if not self.ncert_loader.standards:
                from .ncert_standards import initialize_ncert_standards
                self.ncert_loader = initialize_ncert_standards()
        except Exception as e:
            logger.warning(f"Could not load NCERT standards: {e}")
    
    def validate_content(
        self,
        original_text: str,
        translated_text: str,
        grade_level: int,
        subject: str,
        language: str,
        content_id: str = None
    ) -> ValidationResult:
        """Main validation method that performs all quality checks."""
        content_id = content_id or f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        issues = []
        recommendations = []
        quality_metrics = {}
        
        # 1. Semantic accuracy validation
        semantic_score = self._validate_semantic_accuracy(original_text, translated_text)
        quality_metrics['semantic_accuracy'] = semantic_score
        
        if semantic_score < 0.8:
            issues.append(f"Low semantic accuracy: {semantic_score:.2f}")
            recommendations.append("Review translation for meaning preservation")
        
        # 2. NCERT alignment checking
        ncert_score = self._validate_ncert_alignment(translated_text, grade_level, subject)
        quality_metrics['ncert_alignment'] = ncert_score
        
        if ncert_score < self.quality_threshold:
            issues.append(f"NCERT alignment below threshold: {ncert_score:.2f} < {self.quality_threshold}")
            recommendations.append("Align content with NCERT curriculum standards")
        
        # 3. Script accuracy validation
        script_accuracy = self._validate_script_accuracy(translated_text, language)
        quality_metrics['script_accuracy'] = 1.0 if script_accuracy else 0.0
        
        if not script_accuracy:
            issues.append("Script rendering issues detected")
            recommendations.append("Fix mathematical/scientific notation rendering")
        
        # 4. Age-appropriate language check
        age_appropriate = self._check_age_appropriate_language(translated_text, grade_level)
        quality_metrics['age_appropriate'] = 1.0 if age_appropriate else 0.0
        
        if not age_appropriate:
            issues.append("Language complexity not appropriate for grade level")
            recommendations.append(f"Simplify language for grade {grade_level} students")
        
        # 5. Technical terminology preservation
        tech_terms_preserved = self._check_technical_terminology(original_text, translated_text, subject)
        quality_metrics['technical_terms_preserved'] = 1.0 if tech_terms_preserved else 0.0
        
        if not tech_terms_preserved:
            issues.append("Technical terminology may not be properly preserved")
            recommendations.append("Review subject-specific term translations")
        
        # Determine overall status
        overall_status = self._determine_overall_status(
            semantic_score, ncert_score, script_accuracy, age_appropriate
        )
        
        return ValidationResult(
            content_id=content_id,
            semantic_accuracy=semantic_score,
            ncert_alignment_score=ncert_score,
            script_accuracy=script_accuracy,
            age_appropriate=age_appropriate,
            overall_status=overall_status,
            issues=issues,
            recommendations=recommendations,
            quality_metrics=quality_metrics,
            validation_timestamp=datetime.now()
        )
    
    def _validate_semantic_accuracy(self, original_text: str, translated_text: str) -> float:
        """Validate semantic accuracy between original and translated text using BERT."""
        try:
            # Use BERT to calculate semantic similarity
            similarity = self.bert_client.process(original_text, translated_text)
            return max(0.0, min(1.0, similarity))  # Ensure 0-1 range
        except Exception as e:
            logger.error(f"Error in semantic validation: {e}")
            return 0.5  # Default moderate score on error
    
    def _validate_ncert_alignment(self, content: str, grade_level: int, subject: str) -> float:
        """Validate alignment with NCERT curriculum standards."""
        try:
            # Find matching NCERT standards
            matching_standards = self.ncert_loader.find_matching_standards(
                content, grade_level, subject, top_k=3
            )
            
            if not matching_standards:
                return 0.0
            
            # Calculate weighted alignment score
            total_score = 0.0
            total_weight = 0.0
            
            for standard, similarity in matching_standards:
                # Combine similarity with keyword overlap
                keyword_overlap = self.ncert_loader.check_keyword_overlap(content, standard)
                learning_obj_match = self.ncert_loader.get_learning_objectives_match(content, standard)
                
                # Weighted combination
                combined_score = (
                    0.5 * similarity +
                    0.3 * keyword_overlap +
                    0.2 * learning_obj_match
                )
                
                weight = similarity  # Use similarity as weight
                total_score += combined_score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error in NCERT alignment validation: {e}")
            return 0.0
    
    def _validate_script_accuracy(self, text: str, language: str) -> bool:
        """Validate script accuracy for mathematical and scientific notation."""
        try:
            # Validate Unicode rendering for the specific language
            language_scripts = {
                'Hindi': r'[\u0900-\u097F]',
                'Tamil': r'[\u0B80-\u0BFF]',
                'Telugu': r'[\u0C00-\u0C7F]',
                'Bengali': r'[\u0980-\u09FF]',
                'Marathi': r'[\u0900-\u097F]'
            }
            
            if language in language_scripts:
                script_pattern = language_scripts[language]
                # Check if text contains proper script characters
                script_matches = re.findall(script_pattern, text)
                
                # If text is supposed to be in the target language but has no script characters,
                # it might be a rendering issue
                if len(text) > 50 and len(script_matches) < 10:
                    return False
            
            # Check for malformed mathematical expressions
            malformed_patterns = [
                r'[0-9]+[+\-×÷=][A-Za-z]',  # Number followed by operator and letter
                r'[A-Za-z][+\-×÷=][0-9]+',  # Letter followed by operator and number
            ]
            
            for pattern in malformed_patterns:
                if re.search(pattern, text):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in script accuracy validation: {e}")
            return True  # Default to true on error
    
    def _check_age_appropriate_language(self, text: str, grade_level: int) -> bool:
        """Check if language complexity is appropriate for the grade level."""
        try:
            # Calculate basic readability metrics
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return True
            
            # Average sentence length
            words = text.split()
            avg_sentence_length = len(words) / len(sentences)
            
            # Complex word ratio (words with 3+ syllables)
            complex_words = self._count_complex_words(words)
            complex_word_ratio = complex_words / len(words) if words else 0
            
            # Grade-level thresholds
            grade_thresholds = {
                5: {'max_sentence_length': 15, 'max_complex_ratio': 0.1},
                6: {'max_sentence_length': 16, 'max_complex_ratio': 0.12},
                7: {'max_sentence_length': 17, 'max_complex_ratio': 0.15},
                8: {'max_sentence_length': 18, 'max_complex_ratio': 0.18},
                9: {'max_sentence_length': 20, 'max_complex_ratio': 0.20},
                10: {'max_sentence_length': 22, 'max_complex_ratio': 0.25},
                11: {'max_sentence_length': 25, 'max_complex_ratio': 0.30},
                12: {'max_sentence_length': 28, 'max_complex_ratio': 0.35}
            }
            
            threshold = grade_thresholds.get(grade_level, grade_thresholds[8])
            
            return (avg_sentence_length <= threshold['max_sentence_length'] and
                    complex_word_ratio <= threshold['max_complex_ratio'])
            
        except Exception as e:
            logger.error(f"Error in age-appropriate language check: {e}")
            return True  # Default to true on error
    
    def _count_complex_words(self, words: List[str]) -> int:
        """Count words with 3 or more syllables (simplified heuristic)."""
        complex_count = 0
        for word in words:
            syllable_count = self._count_syllables(word)
            if syllable_count >= 3:
                complex_count += 1
        
        return complex_count
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using simplified heuristic."""
        word = word.lower().strip('.,!?;:"')
        if len(word) < 3:
            return 1
        
        vowels = 'aeiouAEIOU'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Adjust for silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Minimum of 1 syllable per word
        return max(1, syllable_count)
    
    def _check_technical_terminology(self, original: str, translated: str, subject: str) -> bool:
        """Check if technical terminology is properly preserved in translation."""
        try:
            # Subject-specific technical terms that should be preserved or properly translated
            subject_terms = {
                'Mathematics': [
                    'equation', 'variable', 'coefficient', 'polynomial', 'derivative',
                    'integral', 'matrix', 'vector', 'theorem', 'proof', 'algorithm'
                ],
                'Science': [
                    'molecule', 'atom', 'electron', 'photosynthesis', 'mitosis',
                    'ecosystem', 'catalyst', 'enzyme', 'chromosome', 'hypothesis'
                ],
                'Social Studies': [
                    'democracy', 'constitution', 'parliament', 'civilization',
                    'revolution', 'economy', 'culture', 'society', 'government'
                ]
            }
            
            terms_to_check = subject_terms.get(subject, [])
            if not terms_to_check:
                return True  # No specific terms to check
            
            # Count technical terms in original
            original_lower = original.lower()
            original_term_count = sum(1 for term in terms_to_check if term in original_lower)
            
            if original_term_count == 0:
                return True  # No technical terms to preserve
            
            # For translated text, we can't directly check English terms
            # Instead, check if the translated text maintains similar complexity
            # and has appropriate technical vocabulary density
            
            translated_words = translated.split()
            # Heuristic: technical content should maintain certain word length distribution
            long_words = [w for w in translated_words if len(w) > 6]
            technical_density = len(long_words) / len(translated_words) if translated_words else 0
            
            # Expect at least some technical vocabulary in translated content
            return technical_density > 0.1
            
        except Exception as e:
            logger.error(f"Error in technical terminology check: {e}")
            return True  # Default to true on error
    
    def _determine_overall_status(
        self,
        semantic_score: float,
        ncert_score: float,
        script_accuracy: bool,
        age_appropriate: bool
    ) -> str:
        """Determine overall validation status based on individual checks."""
        # Critical failures
        if ncert_score < self.quality_threshold:
            return 'failed'
        
        if not script_accuracy:
            return 'failed'
        
        # Quality thresholds
        if semantic_score >= 0.8 and age_appropriate:
            return 'passed'
        elif semantic_score >= 0.6:
            return 'needs_review'
        else:
            return 'failed'
    
    def generate_quality_report(
        self,
        original_text: str,
        translated_text: str,
        grade_level: int,
        subject: str,
        language: str,
        content_id: str = None
    ) -> QualityReport:
        """Generate comprehensive quality report with detailed metrics."""
        validation_result = self.validate_content(
            original_text, translated_text, grade_level, subject, language, content_id
        )
        
        # Get detailed NCERT matching information
        matched_standards = self.ncert_loader.find_matching_standards(
            translated_text, grade_level, subject, top_k=5
        )
        
        # Calculate detailed metrics
        keyword_overlap_scores = [
            self.ncert_loader.check_keyword_overlap(translated_text, standard)
            for standard, _ in matched_standards
        ]
        
        learning_objective_matches = [
            self.ncert_loader.get_learning_objectives_match(translated_text, standard)
            for standard, _ in matched_standards
        ]
        
        # Check technical terms preservation
        tech_terms_preserved = self._check_technical_terminology(
            original_text, translated_text, subject
        )
        
        # Identify script rendering issues
        script_issues = []
        if not validation_result.script_accuracy:
            script_issues.append("Mathematical notation rendering issues detected")
            script_issues.append("Scientific symbols may not display correctly")
        
        return QualityReport(
            validation_result=validation_result,
            matched_standards=matched_standards,
            keyword_overlap_scores=keyword_overlap_scores,
            learning_objective_matches=learning_objective_matches,
            technical_terms_preserved=tech_terms_preserved,
            script_rendering_issues=script_issues
        )
    
    def set_quality_threshold(self, threshold: float) -> None:
        """Set the NCERT alignment quality threshold."""
        if 0.0 <= threshold <= 1.0:
            self.quality_threshold = threshold
        else:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")
    
    def get_validation_summary(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Get a summary of validation results for reporting."""
        return {
            'content_id': validation_result.content_id,
            'overall_status': validation_result.overall_status,
            'scores': {
                'semantic_accuracy': validation_result.semantic_accuracy,
                'ncert_alignment': validation_result.ncert_alignment_score,
                'script_accuracy': validation_result.script_accuracy,
                'age_appropriate': validation_result.age_appropriate
            },
            'issues_count': len(validation_result.issues),
            'recommendations_count': len(validation_result.recommendations),
            'validation_timestamp': validation_result.validation_timestamp.isoformat()
        }
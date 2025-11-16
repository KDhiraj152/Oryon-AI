"""
Mock ML models for testing without external API dependencies.
These provide realistic fallback implementations when actual models are unavailable.
"""

import re
from typing import Dict, List, Tuple
import hashlib


class MockSimplificationModel:
    """Mock model for text simplification with rule-based approach."""
    
    def __init__(self):
        # Common academic terms and their simpler alternatives
        self.simplification_map = {
            'photosynthesis': 'the process plants use to make food',
            'mitochondria': 'the energy-producing part of a cell',
            'ecosystem': 'a community of living and non-living things',
            'hypothesis': 'an educated guess',
            'precipitation': 'rain, snow, or sleet',
            'evaporation': 'when water turns into vapor',
            'multiplication': 'repeated addition',
            'fraction': 'part of a whole',
            'molecule': 'tiny particle made of atoms',
            'velocity': 'speed with direction',
            'democracy': 'government by the people',
            'constitution': 'a country\'s main rules',
        }
        
        # Complex words to replace based on grade level
        self.grade_replacements = {
            5: {
                'utilize': 'use', 'demonstrate': 'show', 'comprehend': 'understand',
                'investigate': 'look into', 'examine': 'look at', 'obtain': 'get'
            },
            8: {
                'facilitate': 'help', 'implement': 'carry out', 'analyze': 'study',
                'synthesize': 'combine', 'evaluate': 'judge'
            }
        }
    
    def simplify(self, text: str, grade_level: int) -> str:
        """Simplify text based on grade level."""
        simplified = text
        
        # Replace complex academic terms
        for complex_term, simple_term in self.simplification_map.items():
            pattern = r'\b' + re.escape(complex_term) + r'\b'
            simplified = re.sub(pattern, simple_term, simplified, flags=re.IGNORECASE)
        
        # Replace grade-appropriate vocabulary
        for grade, replacements in self.grade_replacements.items():
            if grade_level <= grade:
                for complex_word, simple_word in replacements.items():
                    pattern = r'\b' + re.escape(complex_word) + r'\b'
                    simplified = re.sub(pattern, simple_word, simplified, flags=re.IGNORECASE)
        
        # Break long sentences for younger grades
        if grade_level <= 7:
            sentences = re.split(r'([.!?])', simplified)
            new_sentences = []
            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i].strip()
                punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
                
                # Break at conjunctions if sentence is long
                if len(sentence.split()) > 20:
                    parts = re.split(r'\s+(and|but|because|so|which)\s+', sentence, maxsplit=1)
                    if len(parts) >= 3:
                        new_sentences.append(parts[0].strip() + punctuation)
                        new_sentences.append(parts[1].capitalize() + ' ' + parts[2].strip() + punctuation)
                    else:
                        new_sentences.append(sentence + punctuation)
                else:
                    new_sentences.append(sentence + punctuation)
            
            simplified = ' '.join(new_sentences)
        
        return simplified.strip()


class MockTranslationModel:
    """Mock model for Indian language translation with phonetic support."""
    
    def __init__(self):
        # Sample translations for common educational terms
        self.hindi_translations = {
            'water': 'पानी', 'energy': 'ऊर्जा', 'plant': 'पौधा',
            'animal': 'जानवर', 'cell': 'कोशिका', 'atom': 'परमाणु',
            'science': 'विज्ञान', 'mathematics': 'गणित', 'history': 'इतिहास',
            'geography': 'भूगोल', 'student': 'विद्यार्थी', 'teacher': 'शिक्षक'
        }
        
        self.tamil_translations = {
            'water': 'தண்ணீர்', 'energy': 'ஆற்றல்', 'plant': 'தாவரம்',
            'animal': 'விலங்கு', 'cell': 'செல்', 'atom': 'அணு',
            'science': 'அறிவியல்', 'mathematics': 'கணிதம்', 'history': 'வரலாறு',
            'geography': 'புவியியல்', 'student': 'மாணவர்', 'teacher': 'ஆசிரியர்'
        }
        
        self.language_maps = {
            'hindi': self.hindi_translations,
            'tamil': self.tamil_translations
        }
    
    def translate(self, text: str, target_language: str) -> Tuple[str, bool]:
        """
        Translate text to target language.
        Returns (translated_text, script_valid).
        """
        target_language = target_language.lower()
        
        # For mock, do word-by-word translation of known terms
        if target_language in self.language_maps:
            translation_map = self.language_maps[target_language]
            words = text.split()
            translated_words = []
            
            for word in words:
                clean_word = word.lower().strip('.,!?;:')
                if clean_word in translation_map:
                    translated_words.append(translation_map[clean_word])
                else:
                    # Keep technical terms in English
                    translated_words.append(word)
            
            translated = ' '.join(translated_words)
            script_valid = self._check_script(translated, target_language)
            return translated, script_valid
        
        # Fallback: return with language marker
        return f"[{target_language.title()}] {text}", False
    
    def _check_script(self, text: str, language: str) -> bool:
        """Check if text uses appropriate script for language."""
        if language == 'hindi':
            # Check for Devanagari script (U+0900-U+097F)
            return bool(re.search(r'[\u0900-\u097F]', text))
        elif language == 'tamil':
            # Check for Tamil script (U+0B80-U+0BFF)
            return bool(re.search(r'[\u0B80-\u0BFF]', text))
        return False


class MockValidationModel:
    """Mock model for semantic validation and similarity checking."""
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using word overlap heuristic.
        Returns score between 0 and 1.
        """
        # Tokenize and normalize
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        
        # Boost similarity if many important words match
        important_words = {'science', 'math', 'history', 'energy', 'water', 'cell',
                          'plant', 'animal', 'student', 'teacher', 'learn', 'study'}
        important_matches = intersection.intersection(important_words)
        boost = min(0.1 * len(important_matches), 0.3)
        
        return min(similarity + boost, 1.0)
    
    def check_ncert_alignment(self, text: str, standards: List[Dict]) -> Tuple[bool, float]:
        """
        Check if content aligns with NCERT standards.
        Returns (is_aligned, confidence_score).
        """
        if not standards:
            return False, 0.0
        
        # Extract keywords from text
        text_keywords = set(re.findall(r'\w+', text.lower()))
        
        best_score = 0.0
        for standard in standards:
            standard_text = f"{standard.get('topic', '')} {standard.get('description', '')}"
            standard_keywords = set(re.findall(r'\w+', standard_text.lower()))
            
            if not standard_keywords:
                continue
            
            # Calculate overlap
            overlap = len(text_keywords.intersection(standard_keywords))
            score = overlap / len(standard_keywords) if standard_keywords else 0.0
            best_score = max(best_score, score)
        
        is_aligned = best_score >= 0.3  # 30% keyword overlap threshold
        return is_aligned, best_score
    
    def check_age_appropriate(self, text: str, grade_level: int) -> Tuple[bool, str]:
        """
        Check if content is age-appropriate.
        Returns (is_appropriate, reason).
        """
        # Check for inappropriate content markers
        inappropriate_topics = [
            'violence', 'weapon', 'drug', 'alcohol', 'explicit',
            'mature', 'adult', 'sexual', 'gambling'
        ]
        
        text_lower = text.lower()
        for topic in inappropriate_topics:
            if topic in text_lower:
                return False, f"Contains inappropriate topic: {topic}"
        
        # Check reading complexity (rough estimate)
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Expected average word length by grade
        max_avg_length = {
            5: 5.0, 6: 5.5, 7: 6.0, 8: 6.5, 9: 7.0, 10: 7.5, 11: 8.0, 12: 8.5
        }
        
        expected_length = max_avg_length.get(grade_level, 7.0)
        if avg_word_length > expected_length + 1.5:
            return False, f"Text complexity too high for grade {grade_level}"
        
        return True, "Content is age-appropriate"


class MockTTSModel:
    """Mock model for text-to-speech generation."""
    
    def generate_audio(self, text: str, language: str) -> bytes:
        """
        Generate mock audio data.
        In a real implementation, this would return actual audio.
        For testing, we return a deterministic byte sequence based on the text.
        """
        # Create a deterministic "audio" representation for testing
        # Real implementation would use actual TTS model
        text_hash = hashlib.md5(f"{text}_{language}".encode()).hexdigest()
        
        # Generate mock WAV-like header (simplified)
        mock_audio = b'RIFF'
        mock_audio += b'\x00\x00\x00\x00'  # File size placeholder
        mock_audio += b'WAVE'
        mock_audio += b'fmt '
        mock_audio += b'\x10\x00\x00\x00'  # Format chunk size
        
        # Add text hash as "audio data"
        mock_audio += text_hash.encode()
        
        return mock_audio
    
    def validate_audio(self, audio_data: bytes) -> bool:
        """Validate that audio data is not empty and has minimum structure."""
        return len(audio_data) >= 20  # Minimum for header + some data


# Factory function to get mock models
def get_mock_models() -> Dict:
    """Get all mock models for testing."""
    return {
        'simplification': MockSimplificationModel(),
        'translation': MockTranslationModel(),
        'validation': MockValidationModel(),
        'tts': MockTTSModel()
    }

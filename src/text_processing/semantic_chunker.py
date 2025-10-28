"""
Semantic Chunking Module with LLM-based Text Analysis.

This module provides advanced chunking capabilities using LLM to analyze semantic 
boundaries, topic coherence, and natural text structure for optimal chunk creation.
"""

from typing import List, Dict, Optional, Tuple, Union
import re
import json
from dataclasses import dataclass
from ..utils.cloud_llm_client import get_cloud_llm_client
from ..utils.language_detector import detect_query_language
from ..config import get_config
from ..utils.logger import get_logger

@dataclass
class SemanticBoundary:
    """Represents a semantic boundary point in text."""
    position: int
    confidence: float
    topic_shift: bool
    coherence_score: float
    boundary_type: str  # 'topic_change', 'subtopic_change', 'context_shift'

@dataclass
class TopicSegment:
    """Represents a coherent topic segment."""
    start_pos: int
    end_pos: int
    topic: str
    coherence_score: float
    key_concepts: List[str]
    text: str

class SemanticChunker:
    """LLM-powered semantic text chunker with Turkish language support."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__, self.config)
        self.llm_client = get_cloud_llm_client()
        
        # Default model configuration for Groq
        self.semantic_model = "llama-3.1-8b-instant"
        self.max_analysis_tokens = 2048
        self.min_chunk_size = 100
        self.max_chunk_size = 1000
        self.overlap_strategy = "semantic_aware"
        
    def analyze_semantic_structure(self, text: str, language: str = "auto") -> List[SemanticBoundary]:
        """Analyze text structure using LLM to identify semantic boundaries."""
        
        if language == "auto":
            language = detect_query_language(text)
        
        # Split text into analyzable segments
        segments = self._prepare_segments_for_analysis(text)
        boundaries = []
        
        for i, segment in enumerate(segments):
            if len(segment) < 50:  # Skip very short segments
                continue
                
            try:
                segment_boundaries = self._analyze_segment_boundaries(segment, language)
                # Adjust positions relative to full text
                adjusted_boundaries = []
                segment_start = sum(len(s) for s in segments[:i])
                
                for boundary in segment_boundaries:
                    adjusted_boundary = SemanticBoundary(
                        position=boundary.position + segment_start,
                        confidence=boundary.confidence,
                        topic_shift=boundary.topic_shift,
                        coherence_score=boundary.coherence_score,
                        boundary_type=boundary.boundary_type
                    )
                    adjusted_boundaries.append(adjusted_boundary)
                
                boundaries.extend(adjusted_boundaries)
                
            except Exception as e:
                self.logger.warning(f"Segment analysis failed: {e}")
                continue
        
        return self._merge_and_filter_boundaries(boundaries)
    
    def _prepare_segments_for_analysis(self, text: str) -> List[str]:
        """Prepare text segments for LLM analysis."""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        segments = []
        current_segment = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If adding this paragraph would exceed max tokens, finalize current segment
            if current_length + paragraph_length > self.max_analysis_tokens and current_segment:
                segments.append('\n\n'.join(current_segment))
                current_segment = [paragraph]
                current_length = paragraph_length
            else:
                current_segment.append(paragraph)
                current_length += paragraph_length
        
        if current_segment:
            segments.append('\n\n'.join(current_segment))
        
        return segments
    
    def _analyze_segment_boundaries(self, segment: str, language: str) -> List[SemanticBoundary]:
        """Analyze a single segment for semantic boundaries using LLM."""
        
        # Create analysis prompt based on language
        if language == "tr" or language == "turkish":
            system_prompt = """Sen akademik metinleri analiz eden bir uzmansın. Verilen metindeki anlamsal sınırları ve konu geçişlerini tespit ediyorsun.

Görüşlerin:
1. Konu değişimleri - ana konu değiştiğinde
2. Alt konu geçişleri - aynı ana konu içinde alt konu değiştiğinde  
3. Bağlam değişimleri - perspektif veya yaklaşım değiştiğinde
4. Anlamsal tutarlılık skorları (0.0-1.0)

Yanıtını JSON formatında ver."""
            
            user_prompt = f"""Bu metindeki anlamsal sınırları analiz et:

{segment}

Her potansiyel sınır için şunları belirt:
- position: Karakter pozisyonu
- boundary_type: "topic_change", "subtopic_change", veya "context_shift"
- confidence: Güven skoru (0.0-1.0)
- topic_shift: Ana konu değişimi var mı? (true/false)
- coherence_score: Anlamsal tutarlılık skoru (0.0-1.0)

Yanıt formatı:
{{"boundaries": [...]}}"""
        else:
            system_prompt = """You are an expert in analyzing academic texts for semantic boundaries and topic transitions.

You identify:
1. Topic changes - when the main topic shifts
2. Subtopic transitions - when subtopics change within the same main topic
3. Context shifts - when perspective or approach changes
4. Semantic coherence scores (0.0-1.0)

Respond in JSON format."""
            
            user_prompt = f"""Analyze the semantic boundaries in this text:

{segment}

For each potential boundary, specify:
- position: Character position
- boundary_type: "topic_change", "subtopic_change", or "context_shift"
- confidence: Confidence score (0.0-1.0)
- topic_shift: Is there a main topic change? (true/false)
- coherence_score: Semantic coherence score (0.0-1.0)

Response format:
{{"boundaries": [...]}}"""
        
        try:
            response = self.llm_client.generate(
                model=self.semantic_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Low temperature for consistent analysis
                max_tokens=1024
            )
            
            # Parse JSON response
            boundaries_data = json.loads(response)
            boundaries = []
            
            for boundary_data in boundaries_data.get("boundaries", []):
                boundary = SemanticBoundary(
                    position=boundary_data.get("position", 0),
                    confidence=boundary_data.get("confidence", 0.5),
                    topic_shift=boundary_data.get("topic_shift", False),
                    coherence_score=boundary_data.get("coherence_score", 0.5),
                    boundary_type=boundary_data.get("boundary_type", "context_shift")
                )
                boundaries.append(boundary)
            
            return boundaries
            
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM response as JSON")
            return []
        except Exception as e:
            self.logger.error(f"LLM boundary analysis failed: {e}")
            return []
    
    def _merge_and_filter_boundaries(self, boundaries: List[SemanticBoundary]) -> List[SemanticBoundary]:
        """Merge close boundaries and filter by confidence."""
        if not boundaries:
            return []
        
        # Sort by position
        boundaries.sort(key=lambda b: b.position)
        
        # Merge boundaries that are very close to each other
        merged = []
        min_distance = 50  # Minimum distance between boundaries
        
        for boundary in boundaries:
            if not merged or boundary.position - merged[-1].position >= min_distance:
                merged.append(boundary)
            else:
                # Merge with previous boundary - keep higher confidence
                prev = merged[-1]
                if boundary.confidence > prev.confidence:
                    merged[-1] = boundary
        
        # Filter by confidence threshold
        confidence_threshold = 0.6
        filtered = [b for b in merged if b.confidence >= confidence_threshold]
        
        return filtered
    
    def identify_topic_segments(self, text: str, boundaries: List[SemanticBoundary]) -> List[TopicSegment]:
        """Identify coherent topic segments using semantic boundaries."""
        
        if not boundaries:
            # If no boundaries found, treat entire text as one segment
            return [TopicSegment(
                start_pos=0,
                end_pos=len(text),
                topic="Unknown",
                coherence_score=0.7,
                key_concepts=[],
                text=text
            )]
        
        segments = []
        start_pos = 0
        
        for boundary in boundaries:
            if boundary.position > start_pos:
                segment_text = text[start_pos:boundary.position].strip()
                if len(segment_text) >= self.min_chunk_size:
                    topic_info = self._analyze_topic_coherence(segment_text)
                    
                    segment = TopicSegment(
                        start_pos=start_pos,
                        end_pos=boundary.position,
                        topic=topic_info.get("topic", "Unknown"),
                        coherence_score=topic_info.get("coherence_score", boundary.coherence_score),
                        key_concepts=topic_info.get("key_concepts", []),
                        text=segment_text
                    )
                    segments.append(segment)
                
                start_pos = boundary.position
        
        # Add final segment
        if start_pos < len(text):
            segment_text = text[start_pos:].strip()
            if len(segment_text) >= self.min_chunk_size:
                topic_info = self._analyze_topic_coherence(segment_text)
                
                segment = TopicSegment(
                    start_pos=start_pos,
                    end_pos=len(text),
                    topic=topic_info.get("topic", "Unknown"),
                    coherence_score=topic_info.get("coherence_score", 0.7),
                    key_concepts=topic_info.get("key_concepts", []),
                    text=segment_text
                )
                segments.append(segment)
        
        return segments
    
    def _analyze_topic_coherence(self, text: str) -> Dict[str, Union[str, float, List[str]]]:
        """Analyze topic coherence and extract key concepts."""
        
        if len(text) < 100:
            return {"topic": "Short segment", "coherence_score": 0.8, "key_concepts": []}
        
        # Simple heuristic-based analysis to avoid excessive LLM calls
        sentences = text.split('.')
        if len(sentences) <= 3:
            return {"topic": "Brief topic", "coherence_score": 0.9, "key_concepts": []}
        
        # Extract key terms using simple frequency analysis
        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top concepts
        key_concepts = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        # Simple coherence estimation based on text structure
        coherence_score = min(0.9, 0.5 + len(key_concepts) * 0.08)
        
        # Generate topic label from key concepts
        topic = f"Topic: {', '.join(key_concepts[:2])}" if key_concepts else "General content"
        
        return {
            "topic": topic,
            "coherence_score": coherence_score,
            "key_concepts": key_concepts
        }
    
    def create_semantic_chunks(
        self, 
        text: str, 
        target_size: int = 800,
        overlap_ratio: float = 0.1,
        language: str = "auto"
    ) -> List[str]:
        """Create semantically coherent chunks with intelligent overlap."""
        
        if not text.strip():
            return []
        
        self.logger.info(f"Creating semantic chunks for text of length {len(text)}")
        
        # Step 1: Analyze semantic structure
        boundaries = self.analyze_semantic_structure(text, language)
        self.logger.info(f"Found {len(boundaries)} semantic boundaries")
        
        # Step 2: Identify topic segments
        segments = self.identify_topic_segments(text, boundaries)
        self.logger.info(f"Identified {len(segments)} topic segments")
        
        # Step 3: Create chunks with adaptive sizing
        chunks = self._create_adaptive_chunks(segments, target_size, overlap_ratio)
        
        # Step 4: Apply semantic-aware overlap
        if overlap_ratio > 0:
            chunks = self._apply_semantic_overlap(chunks, text, overlap_ratio)
        
        self.logger.info(f"Generated {len(chunks)} semantic chunks")
        return chunks
    
    def _create_adaptive_chunks(
        self, 
        segments: List[TopicSegment], 
        target_size: int, 
        overlap_ratio: float
    ) -> List[str]:
        """Create adaptively-sized chunks from topic segments."""
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for segment in segments:
            segment_size = len(segment.text)
            
            # If segment alone is too large, split it
            if segment_size > self.max_chunk_size:
                # Finalize current chunk if exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split large segment
                sub_chunks = self._split_large_segment(segment, target_size)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this segment would exceed target size
            if current_size + segment_size > target_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [segment.text]
                current_size = segment_size
            else:
                current_chunk.append(segment.text)
                current_size += segment_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_large_segment(self, segment: TopicSegment, target_size: int) -> List[str]:
        """Split a large segment while preserving semantic coherence."""
        
        text = segment.text
        if len(text) <= target_size:
            return [text]
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            # If only one paragraph, split by sentences
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            return self._group_by_size(sentences, target_size)
        else:
            return self._group_by_size(paragraphs, target_size)
    
    def _group_by_size(self, units: List[str], target_size: int) -> List[str]:
        """Group text units by target size."""
        
        groups = []
        current_group = []
        current_size = 0
        
        for unit in units:
            unit_size = len(unit)
            
            if current_size + unit_size > target_size and current_group:
                groups.append(' '.join(current_group))
                current_group = [unit]
                current_size = unit_size
            else:
                current_group.append(unit)
                current_size += unit_size
        
        if current_group:
            groups.append(' '.join(current_group))
        
        return groups
    
    def _apply_semantic_overlap(self, chunks: List[str], original_text: str, overlap_ratio: float) -> List[str]:
        """Apply semantic-aware overlap between chunks."""
        
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i-1]
            
            # Calculate overlap size
            overlap_size = int(len(previous_chunk) * overlap_ratio)
            
            # Get semantic overlap from previous chunk
            overlap_text = self._get_semantic_overlap(previous_chunk, overlap_size)
            
            if overlap_text:
                # Combine overlap with current chunk
                combined_chunk = overlap_text + " " + current_chunk
                overlapped_chunks.append(combined_chunk.strip())
            else:
                overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def _get_semantic_overlap(self, text: str, target_size: int) -> str:
        """Extract semantically meaningful overlap from end of text."""
        
        if len(text) <= target_size:
            return text
        
        # Try to get complete sentences from the end
        sentences = text.split('.')
        if len(sentences) > 1:
            overlap_sentences = []
            current_length = 0
            
            for sentence in reversed(sentences[:-1]):  # Exclude the last empty split
                sentence_with_period = sentence.strip() + '.'
                if current_length + len(sentence_with_period) <= target_size:
                    overlap_sentences.insert(0, sentence_with_period)
                    current_length += len(sentence_with_period)
                else:
                    break
            
            if overlap_sentences:
                return ' '.join(overlap_sentences)
        
        # Fallback to word-based overlap
        words = text.split()
        if len(words) > 10:
            overlap_words = []
            current_length = 0
            
            for word in reversed(words):
                if current_length + len(word) + 1 <= target_size:
                    overlap_words.insert(0, word)
                    current_length += len(word) + 1
                else:
                    break
            
            return ' '.join(overlap_words)
        
        # Last resort: character-based overlap
        return text[-target_size:]


def create_semantic_chunks(
    text: str,
    target_size: int = 800,
    overlap_ratio: float = 0.1,
    language: str = "auto",
    fallback_strategy: str = "markdown"
) -> List[str]:
    """
    Main function to create semantic chunks with fallback support.
    
    Args:
        text: Input text to chunk
        target_size: Target size for chunks
        overlap_ratio: Ratio of overlap between chunks  
        language: Language of the text ("tr", "en", or "auto")
        fallback_strategy: Strategy to use if semantic chunking fails
    
    Returns:
        List of semantically coherent text chunks
    """
    chunker = SemanticChunker()
    
    try:
        return chunker.create_semantic_chunks(text, target_size, overlap_ratio, language)
    except Exception as e:
        chunker.logger.error(f"Semantic chunking failed: {e}")
        chunker.logger.info(f"Falling back to {fallback_strategy} strategy")
        
        # Import and use fallback strategy
        from .text_chunker import chunk_text
        return chunk_text(
            text=text,
            chunk_size=target_size,
            chunk_overlap=int(target_size * overlap_ratio),
            strategy=fallback_strategy
        )
"""
Embedding generation module.

This module provides functions to generate text embeddings using multiple providers:
- Local Ollama instance
- Cloud-based embeddings (OpenAI, HuggingFace, etc.)
"""

from typing import List, Optional
import hashlib
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .. import config
from ..config import is_cloud_environment
from ..utils.helpers import setup_logging
from ..utils.cache import get_cache

# Optional import for ollama - handle gracefully when not available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

# Optional imports for cloud embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = setup_logging()

# Global clients - initialized on demand
ollama_client = None
sentence_transformer_model = None

def get_selected_provider() -> str:
    """Get the currently selected provider from environment (default: 'groq')."""
    import os
    return os.environ.get("RAG_PROVIDER", "groq")

def init_ollama_client():
    """Initialize Ollama client on demand, skipping in cloud environments."""
    global ollama_client
    if ollama_client is not None:
        return ollama_client

    if is_cloud_environment():
        logger.info("Cloud environment detected, skipping Ollama client initialization.")
        return None
        
    if not OLLAMA_AVAILABLE:
        logger.warning("Ollama module not available")
        return None
        
    try:
        ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        # Test the connection to ensure the server is available
        ollama_client.list()
        logger.info(f"Successfully connected to Ollama at {config.OLLAMA_BASE_URL}")
        return ollama_client
    except Exception as e:
        logger.warning(f"Could not connect to Ollama. Running without local models. Error: {e}")
        ollama_client = None
        return None

def init_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Initialize SentenceTransformer model on demand."""
    global sentence_transformer_model
    if sentence_transformer_model is not None:
        return sentence_transformer_model
        
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("SentenceTransformers not available - install with: pip install sentence-transformers")
        return None
        
    try:
        sentence_transformer_model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded SentenceTransformer model: {model_name}")
        return sentence_transformer_model
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
        sentence_transformer_model = None
        return None

def _truncate_text_for_embedding(text: str, max_length: int = 1500) -> str:
    """
    Truncate text to fit within embedding model context length.
    
    Args:
        text: Input text
        max_length: Maximum character length (approximate token limit)
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    last_sentence_end = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?')
    )
    
    if last_sentence_end > max_length * 0.7:  # If we found a good break point
        return truncated[:last_sentence_end + 1].strip()
    else:
        # Fallback to word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:
            return truncated[:last_space].strip()
        else:
            # Hard truncation as last resort
            return truncated.strip()

def _generate_single_embedding(text: str, model: str) -> List[float]:
    """
    Generate embedding for a single text with length validation.
    """
    client = init_ollama_client()
    if not client:
        return []
    
    # Truncate text if it's too long for the embedding model
    truncated_text = _truncate_text_for_embedding(text)
    
    if len(truncated_text) != len(text):
        logger.warning(f"Text truncated from {len(text)} to {len(truncated_text)} characters for embedding")
    
    try:
        response = client.embed(model=model, input=truncated_text)
        
        # Parse response
        emb = None
        if isinstance(response, dict):
            if 'embedding' in response and isinstance(response['embedding'], list):
                emb = response['embedding']
            elif 'embeddings' in response and isinstance(response['embeddings'], list):
                if response['embeddings'] and isinstance(response['embeddings'][0], list):
                    emb = response['embeddings'][0]
        else:
            # Try attribute access for typed responses
            attr_embedding = getattr(response, 'embedding', None)
            attr_embeddings = getattr(response, 'embeddings', None)
            if isinstance(attr_embedding, list):
                emb = attr_embedding
            elif isinstance(attr_embeddings, list):
                if attr_embeddings and isinstance(attr_embeddings[0], list):
                    emb = attr_embeddings[0]
        
        return emb if emb is not None else []
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

def _get_cache_key(text: str, model: str) -> str:
    """Generate cache key for text and model combination."""
    content = f"{model}:{text}"
    return f"embedding:{hashlib.md5(content.encode()).hexdigest()}"

def generate_embeddings(texts: List[str], model: str = None, use_cache: bool = True, provider: str = None) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using multiple providers.
    Supports caching and batch processing for improved performance.

    Args:
        texts: A list of strings to be embedded.
        model: The embedding model to use. If None, uses the default from config.
        use_cache: Whether to use caching for embeddings.
        provider: Override the provider ('ollama', 'sentence_transformers', 'simple').
                 If None, determines based on selected provider and availability.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
        Returns an empty list if an error occurs or no provider is available.
    """
    if not texts:
        return []
        
    # Determine which provider to use
    if provider is None:
        selected_provider = get_selected_provider()
        if selected_provider == 'ollama':
            # Try Ollama first
            if init_ollama_client() is not None:
                provider = 'ollama'
            else:
                # Fallback to sentence transformers
                provider = 'sentence_transformers'
        else:
            # For cloud providers, use sentence transformers for embeddings
            provider = 'sentence_transformers'
    
    logger.info(f"Generating embeddings for {len(texts)} text(s) using provider '{provider}'")
    
    # Try the requested provider first, then fallback
    if provider == 'ollama':
        result = _generate_embeddings_ollama(texts, model, use_cache)
        if not result and selected_provider != 'ollama':
            # Fallback to sentence transformers if Ollama fails and we're not specifically using Ollama
            logger.warning("Ollama embeddings failed, trying sentence transformers fallback")
            result = _generate_embeddings_sentence_transformers(texts, model, use_cache)
        if not result:
            # Last resort: simple embeddings
            logger.warning("All embedding methods failed, using simple fallback")
            result = _generate_simple_embeddings(texts)
        return result
    elif provider == 'sentence_transformers':
        result = _generate_embeddings_sentence_transformers(texts, model, use_cache)
        if not result:
            # Fallback to simple embeddings if sentence transformers fails
            logger.warning("SentenceTransformers failed, using simple fallback")
            result = _generate_simple_embeddings(texts)
        return result
    else:
        logger.error(f"Unknown embedding provider: {provider}")
        return _generate_simple_embeddings(texts)

def _generate_embeddings_ollama(texts: List[str], model: str = None, use_cache: bool = True) -> List[List[float]]:
    """Generate embeddings using Ollama."""
    client = init_ollama_client()
    if not client:
        logger.error("Cannot generate embeddings: Ollama client is not available.")
        return []

    if model is None:
        model = config.OLLAMA_EMBEDDING_MODEL

    logger.info(f"Generating embeddings using Ollama model '{model}'")
    
    # Gelişmiş text cleaning - Türkçe uyumlu
    cleaned_texts = []
    for text in texts:
        # Temel temizlik
        clean_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
        
        # Çoklu boşlukları tek boşluğa dönüştür
        import re
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Türkçe karakterleri koru, sadece özel karakterleri temizle
        clean_text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?:;()-]', ' ', clean_text)
        
        # Tekrar çoklu boşluk temizliği
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        cleaned_texts.append(clean_text)
    
    # Initialize cache if enabled
    cache = get_cache(ttl=config.CACHE_TTL) if use_cache and config.ENABLE_CACHE else None
    
    embeddings = []
    texts_to_process = []
    indices_to_process = []
    
    # Check cache first
    for i, text in enumerate(cleaned_texts):
        if cache:
            cache_key = _get_cache_key(text, model)
            cached_embedding = cache.get(cache_key)
            if cached_embedding:
                embeddings.append(cached_embedding)
                logger.debug(f"Found cached embedding for text {i}")
                continue
        
        # Need to process this text
        embeddings.append(None)  # Placeholder
        texts_to_process.append(text)
        indices_to_process.append(i)
    
    # Process uncached texts
    if texts_to_process:
        logger.info(f"Processing {len(texts_to_process)} uncached texts")
        
        # Use batch processing with ThreadPoolExecutor for better performance
        batch_size = min(len(texts_to_process), config.EMBEDDING_BATCH_SIZE)
        max_workers = min(batch_size, config.MAX_CONCURRENT_REQUESTS)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, text in enumerate(texts_to_process):
                future = executor.submit(_generate_single_embedding, text, model)
                future_to_index[future] = i
            
            # Collect results
            processed_embeddings = [None] * len(texts_to_process)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embedding = future.result(timeout=config.OLLAMA_REQUEST_TIMEOUT)
                    if embedding:
                        processed_embeddings[index] = embedding
                        
                        # Cache the result
                        if cache:
                            cache_key = _get_cache_key(texts_to_process[index], model)
                            cache.set(cache_key, embedding)
                except Exception as e:
                    logger.error(f"Error processing text {index}: {e}")
                    processed_embeddings[index] = []
            
            # Fill in the results
            for i, embedding in enumerate(processed_embeddings):
                original_index = indices_to_process[i]
                embeddings[original_index] = embedding if embedding else []
    
    # Filter out any None values (shouldn't happen, but safety check)
    final_embeddings = [emb for emb in embeddings if emb is not None]
    
    logger.info(f"Successfully generated {len(final_embeddings)} embeddings with Ollama")
    return final_embeddings

def _generate_embeddings_sentence_transformers(texts: List[str], model: str = None, use_cache: bool = True) -> List[List[float]]:
    """Generate embeddings using SentenceTransformers (local)."""
    transformer_model = init_sentence_transformer(model or "all-MiniLM-L6-v2")
    if not transformer_model:
        logger.error("Cannot generate embeddings: SentenceTransformer not available.")
        return []
    
    logger.info(f"Generating embeddings using SentenceTransformer")
    
    # Clean texts (similar to Ollama version)
    cleaned_texts = []
    for text in texts:
        # Basic cleaning
        clean_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
        
        # Multiple spaces to single space
        import re
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Keep Turkish characters, clean special characters
        clean_text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?:;()-]', ' ', clean_text)
        
        # Clean multiple spaces again
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Truncate if too long (SentenceTransformers has token limits)
        if len(clean_text) > 512:  # Conservative limit
            clean_text = clean_text[:512]
        
        cleaned_texts.append(clean_text)
    
    # Initialize cache if enabled
    cache = get_cache(ttl=config.CACHE_TTL) if use_cache and config.ENABLE_CACHE else None
    
    embeddings = []
    texts_to_process = []
    indices_to_process = []
    
    # Check cache first
    for i, text in enumerate(cleaned_texts):
        if cache:
            cache_key = f"st_embedding:{hashlib.md5(text.encode()).hexdigest()}"
            cached_embedding = cache.get(cache_key)
            if cached_embedding:
                embeddings.append(cached_embedding)
                continue
        
        # Need to process this text
        embeddings.append(None)  # Placeholder
        texts_to_process.append(text)
        indices_to_process.append(i)
    
    # Process uncached texts
    if texts_to_process:
        try:
            # Generate embeddings in batch
            batch_embeddings = transformer_model.encode(texts_to_process, convert_to_numpy=True)
            
            # Convert to list format and cache
            for i, embedding in enumerate(batch_embeddings):
                embedding_list = embedding.tolist()
                original_index = indices_to_process[i]
                embeddings[original_index] = embedding_list
                
                # Cache the result
                if cache:
                    cache_key = f"st_embedding:{hashlib.md5(texts_to_process[i].encode()).hexdigest()}"
                    cache.set(cache_key, embedding_list)
                    
        except Exception as e:
            logger.error(f"Error generating embeddings with SentenceTransformer: {e}")
            return []
    
    # Filter out any None values
    final_embeddings = [emb for emb in embeddings if emb is not None]
    
    logger.info(f"Successfully generated {len(final_embeddings)} embeddings with SentenceTransformer")
    return final_embeddings

def _generate_simple_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate simple hash-based embeddings as a last resort fallback.
    This is NOT suitable for production but prevents complete failure.
    """
    logger.warning("Using simple hash-based embeddings - NOT recommended for production!")
    embeddings = []
    
    for text in texts:
        # Create a simple deterministic embedding based on text hash and content
        import hashlib
        
        # Generate multiple hash values to create a vector
        hash_md5 = hashlib.md5(text.encode()).hexdigest()
        hash_sha1 = hashlib.sha1(text.encode()).hexdigest()
        hash_sha256 = hashlib.sha256(text.encode()).hexdigest()
        
        # Convert hex to numbers and normalize
        embedding = []
        for i in range(0, min(len(hash_md5), 32), 2):
            val = int(hash_md5[i:i+2], 16) / 255.0  # Normalize to [0,1]
            embedding.append(val)
        
        for i in range(0, min(len(hash_sha1), 32), 2):
            val = int(hash_sha1[i:i+2], 16) / 255.0
            embedding.append(val)
            
        # Add some text-based features
        embedding.extend([
            len(text) / 1000.0,  # Text length feature
            text.count(' ') / 100.0,  # Word count feature
            text.count('.') / 10.0,   # Sentence count feature
            text.count('?') / 10.0,   # Question count feature
        ])
        
        # Pad or truncate to fixed size
        target_size = 384  # Common embedding size
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
            
        embeddings.append(embedding)
    
    logger.info(f"Generated {len(embeddings)} simple fallback embeddings")
    return embeddings

if __name__ == '__main__':
    # This is for testing purposes.
    # It requires a running Ollama instance with the specified model pulled.
    # Example: `ollama pull mxbai-embed-large`

    if not client:
        print("Skipping embedding generation test: Ollama client is not available.")
    else:
        print("--- Testing Embedding Generation ---")
        sample_texts = [
            "This is the first test sentence.",
            "Here is another sentence for embedding.",
            "RAG systems rely on high-quality embeddings."
        ]
        
        embeddings = generate_embeddings(sample_texts)
        
        if embeddings:
            print(f"Successfully generated {len(embeddings)} embeddings.")
            for i, emb in enumerate(embeddings):
                print(f"  - Embedding {i+1}: Dimension = {len(emb)}, First 5 values = {emb[:5]}")
            
            # Test single text embedding
            single_embedding = generate_embeddings(["A single piece of text."])
            if single_embedding:
                print("\nSuccessfully generated embedding for a single text.")
                # The result is a list containing one embedding
                print(f"  - Dimension = {len(single_embedding)}")
            else:
                print("\nFailed to generate embedding for a single text.")

        else:
            print("Failed to generate embeddings. Check your Ollama instance and model.")

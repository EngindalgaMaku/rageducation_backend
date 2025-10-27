"""
Configuration management for the RAG system.

This module handles loading and providing access to configuration settings,
including API keys and other parameters, using environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- LLM Provider Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # ollama, huggingface, together, groq, sambanova

# --- Embedding Provider Configuration ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")  # ollama, sentence_transformers
DEFAULT_SENTENCE_TRANSFORMER_MODEL = os.getenv("DEFAULT_SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# --- Ollama API Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "mistral:7b")

# --- Cloud API Configuration ---
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY", "")

# --- Marker PDF Configuration ---
# Force Marker to use only local models (disable Google Gemini)
MARKER_DISABLE_GEMINI = os.getenv("MARKER_DISABLE_GEMINI", "true")
MARKER_USE_LOCAL_ONLY = os.getenv("MARKER_USE_LOCAL_ONLY", "true")
MARKER_LLM_PROVIDER = os.getenv("MARKER_LLM_PROVIDER", "ollama")
MARKER_DISABLE_CLOUD_SERVICES = os.getenv("MARKER_DISABLE_CLOUD_SERVICES", "true")

# Explicitly disable Google services
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- Available Models ---
AVAILABLE_MODELS = {
    # Yerel modeller
    "qwen2.5:14b": {
        "name": "Qwen 2.5 14B",
        "description": "Alibaba'nın güçlü çok dilli modeli - Türkçe desteği mükemmel",
        "size": "~9GB",
        "performance": "High",
        "language": "Multilingual (TR/EN optimized)"
    },
    "mistral:7b": {
        "name": "Mistral 7B",
        "description": "Hızlı ve etkili Fransız yapımı model",
        "size": "~4.4GB",
        "performance": "High",
        "language": "Multilingual (EN optimized)"
    },
    "llama3:8b": {
        "name": "Llama 3 8B",
        "description": "Meta'nın popüler açık kaynak modeli",
        "size": "~4.7GB",
        "performance": "Medium-High",
        "language": "Multilingual"
    },
    
    # Ollama Cloud modeller
    "gpt-oss:120b-cloud": {
        "name": "GPT-OSS 120B (Cloud)",
        "description": "🌐 Açık kaynaklı GPT-style model - Cloud üzerinden",
        "size": "Cloud",
        "performance": "Very High",
        "language": "Multilingual (Strong TR)"
    },
    "gpt-oss:20b-cloud": {
        "name": "GPT-OSS 20B (Cloud)",
        "description": "🌐 Orta ölçekli GPT model - Hızlı cloud erişim",
        "size": "Cloud",
        "performance": "High",
        "language": "Multilingual (Good TR)"
    },
    "deepseek-v3.1:671b-cloud": {
        "name": "DeepSeek V3.1 671B (Cloud)",
        "description": "🌐 Dev ölçekli coding model - Programlama ve genel görevler",
        "size": "Cloud",
        "performance": "Extreme",
        "language": "Multilingual (Excellent TR)"
    },
    "qwen3-coder:480b-cloud": {
        "name": "Qwen3 Coder 480B (Cloud)",
        "description": "🌐 Alibaba'nın coding modeli - Programlama odaklı",
        "size": "Cloud",
        "performance": "Extreme",
        "language": "Multilingual (Perfect TR)"
    },
    "qwen3-vl:235b-cloud": {
        "name": "Qwen3 VL 235B (Cloud)",
        "description": "🌐 Vision-Language model - Görsel + metin anlayışı",
        "size": "Cloud",
        "performance": "Very High",
        "language": "Multilingual (Excellent TR)"
    },
    "glm-4.6:cloud": {
        "name": "GLM-4.6 (Cloud)",
        "description": "🌐 ChatGLM ailesi - Çin yapımı güçlü model",
        "size": "Cloud",
        "performance": "High",
        "language": "Multilingual (Good TR)"
    }
}

# --- Cloud Provider Models --- (Türkçe performansına göre sıralanmış)
CLOUD_MODELS = {
    # Groq Models (En iyi Türkçe performansı)
    "llama-3.1-70b-versatile": {
        "provider": "groq",
        "name": "Llama 3.1 70B (ÖNERİLEN)",
        "description": "🥇 Türkçe'de en başarılı - Akıcı ve doğal dil kullanır",
        "context_length": 131072,
        "turkish_score": 95,
        "free": True
    },
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "name": "Llama 3.1 8B Instant (HIZLI)",
        "description": "🚀 Hızlı ve etkili - Türkçe'de iyi performans",
        "context_length": 131072,
        "turkish_score": 88,
        "free": True
    },
    "mixtral-8x7b-32768": {
        "provider": "groq",
        "name": "Mixtral 8x7B (DENGELİ)",
        "description": "⚖️ Dengeli performans - Türkçe ve İngilizce'de tutarlı",
        "context_length": 32768,
        "turkish_score": 82,
        "free": True
    },
    
    # Hugging Face Models (Orta performans)
    "microsoft/DialoGPT-medium": {
        "provider": "huggingface",
        "name": "DialoGPT Medium",
        "description": "💬 Sohbet odaklı - Türkçe'de orta performans",
        "context_length": 1024,
        "turkish_score": 65,
        "free": True
    },
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "provider": "huggingface",
        "name": "Mistral 7B (HF)",
        "description": "🔧 Instruction-tuned - Basit Türkçe görevler için",
        "context_length": 32768,
        "turkish_score": 72,
        "free": True
    },
    
    # Together AI Models (Alternatif seçenekler)
    "meta-llama/Llama-2-7b-chat-hf": {
        "provider": "together",
        "name": "Llama 2 7B (Together)",
        "description": "🔄 Together AI üzerinden - Türkçe orta düzey",
        "context_length": 4096,
        "turkish_score": 68,
        "free": True
    }
}

# Default model if none specified
DEFAULT_MODEL = "mistral:7b"

# --- GPU Optimization ---
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "1"))
OLLAMA_GPU_LAYERS = int(os.getenv("OLLAMA_GPU_LAYERS", "-1"))
OLLAMA_LOAD_TIMEOUT = int(os.getenv("OLLAMA_LOAD_TIMEOUT", "300"))
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"))

# --- Performance Settings ---
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TOP_K = int(os.getenv("TOP_K", "40"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.1"))

# --- Concurrency Settings ---
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

# --- Cache Settings ---
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# --- Embedding Configuration ---
# Note: The embedding model is now defined in the Ollama section.
# This variable is kept for compatibility but should be phased out.
EMBEDDING_MODEL = OLLAMA_EMBEDDING_MODEL

# --- Text Chunking Configuration ---
# Optimized chunk size for better context coherence
CHUNK_SIZE = 800  # Increased for better text coherence
CHUNK_OVERLAP = 150  # Increased overlap for context continuity
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "sentence")  # Default to sentence-based chunking for better coherence

# --- RAG Strategy Configuration ---
USE_MULTI_QUERY = os.getenv("USE_MULTI_QUERY", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
# Number of documents to retrieve before re-ranking. Should be larger than final top_k.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "25"))

# --- Vector Store Configuration ---
DB_PATH = os.getenv("DB_PATH", "data/analytics/experiments.db")
VECTOR_STORE_PATH = "data/vector_db/faiss_index"

# --- Data Paths ---
RAW_DATA_PATH = "data/raw/sample_documents"
PROCESSED_DATA_PATH = "data/processed"

# --- Logging Configuration ---
LOG_LEVEL = "INFO"
LOG_FILE = "logs/application.log"

def get_config():
    """
    Returns a dictionary containing the main configuration settings.
    This can be expanded to include more complex configuration logic.
    """
    return {
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_embedding_model": OLLAMA_EMBEDDING_MODEL,
        "ollama_generation_model": OLLAMA_GENERATION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_store_path": VECTOR_STORE_PATH,
        "raw_data_path": RAW_DATA_PATH,
        "processed_data_path": PROCESSED_DATA_PATH,
        "log_level": LOG_LEVEL,
        "log_file": LOG_FILE,
        # GPU and Performance Settings
        "ollama_num_gpu": OLLAMA_NUM_GPU,
        "ollama_gpu_layers": OLLAMA_GPU_LAYERS,
        "ollama_load_timeout": OLLAMA_LOAD_TIMEOUT,
        "ollama_request_timeout": OLLAMA_REQUEST_TIMEOUT,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "repeat_penalty": REPEAT_PENALTY,
        # Concurrency Settings
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        # Cache Settings
        "enable_cache": ENABLE_CACHE,
        "cache_ttl": CACHE_TTL,
        # RAG Strategy Settings
        "use_multi_query": USE_MULTI_QUERY,
        "enable_reranking": ENABLE_RERANKING,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        # Available Models
        "available_models": AVAILABLE_MODELS,
        "cloud_models": CLOUD_MODELS,
        "default_model": DEFAULT_MODEL,
        # Cloud API Keys
        "groq_api_key": GROQ_API_KEY,
        "huggingface_api_key": HUGGINGFACE_API_KEY,
        "together_api_key": TOGETHER_API_KEY,
        # Marker PDF Configuration
        "marker_disable_gemini": MARKER_DISABLE_GEMINI,
        "marker_use_local_only": MARKER_USE_LOCAL_ONLY,
        "marker_llm_provider": MARKER_LLM_PROVIDER,
        "marker_disable_cloud_services": MARKER_DISABLE_CLOUD_SERVICES,
        "gemini_api_key": GEMINI_API_KEY,
        "google_api_key": GOOGLE_API_KEY,
        # Embedding Configuration
        "embedding_provider": EMBEDDING_PROVIDER,
        "default_sentence_transformer_model": DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    }

def get_available_models():
    """Get list of available models with their information."""
    return AVAILABLE_MODELS

def set_generation_model(model_name: str):
    """Set the generation model for the current session."""
    import os
    os.environ["OLLAMA_GENERATION_MODEL"] = model_name
    global OLLAMA_GENERATION_MODEL
    OLLAMA_GENERATION_MODEL = model_name

if __name__ == "__main__":
    # Print the configuration for verification
    config = get_config()
    print("--- System Configuration ---")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("--------------------------")

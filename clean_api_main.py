from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Clean RAG3 API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sadece GROQ modelleri (çünkü sadece GROQ_API_KEY var)
GROQ_MODELS = {
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
    "llama3-70b-8192": {
        "provider": "groq",
        "name": "Llama 3 70B",
        "description": "🔥 Meta'nın güçlü 70B modeli",
        "context_length": 8192,
        "turkish_score": 90,
        "free": True
    },
    "llama3-8b-8192": {
        "provider": "groq",
        "name": "Llama 3 8B",
        "description": "⚡ Hızlı ve verimli 8B model",
        "context_length": 8192,
        "turkish_score": 85,
        "free": True
    },
    "gemma-7b-it": {
        "provider": "groq",
        "name": "Gemma 7B IT",
        "description": "🤖 Google'ın instruction-tuned modeli",
        "context_length": 8192,
        "turkish_score": 80,
        "free": True
    }
}

class ListModelsResponse(BaseModel):
    models: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models/list", response_model=ListModelsResponse)
def list_available_models():
    """Get only GROQ models (only provider with API key configured)"""
    return ListModelsResponse(models=list(GROQ_MODELS.keys()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
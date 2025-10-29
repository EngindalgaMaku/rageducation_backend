import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from groq import Groq

# --- Configuration ---
# In a real-world scenario, these would come from a config file or environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Model Inference Service",
    description="A microservice to interact with various LLM providers like Ollama and Groq.",
    version="1.0.0"
)

# --- Pydantic Models for API requests ---
class GenerationRequest(BaseModel):
    prompt: str
    model: str
    # Optional parameters
    temperature: float = 0.7
    max_tokens: int = 1024

class GenerationResponse(BaseModel):
    response: str
    model_used: str

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model_used: str

# --- LLM Clients ---
try:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    # A quick check to see if Ollama is available
    ollama_client.list()
    OLLAMA_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: Could not connect to Ollama at {OLLAMA_HOST}. Ollama models will not be available. Error: {e}")
    ollama_client = None
    OLLAMA_AVAILABLE = False

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if not groq_client:
    print("⚠️ Warning: GROQ_API_KEY not set. Groq models will not be available.")


# --- Helper Functions ---
def is_groq_model(model_name: str) -> bool:
    """Check if the model is intended for Groq based on common naming conventions."""
    groq_models = ["llama-3.1", "mixtral", "gemma"]
    return any(groq_model in model_name for groq_model in groq_models)

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check():
    """Check the operational status of the service and its connections."""
    return {
        "status": "ok",
        "ollama_available": OLLAMA_AVAILABLE,
        "groq_available": bool(groq_client)
    }

@app.post("/models/generate", response_model=GenerationResponse, summary="Generate Response from a Model")
async def generate_response(request: GenerationRequest):
    """
    Receives a prompt and a model name, and returns a generated response.
    It dynamically selects the provider (Ollama or Groq) based on the model name.
    """
    model_name = request.model
    prompt = request.prompt

    try:
        if is_groq_model(model_name):
            if not groq_client:
                raise HTTPException(status_code=503, detail="Groq client is not available. Check GROQ_API_KEY.")

            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            response_content = chat_completion.choices.message.content
            return GenerationResponse(response=response_content, model_used=model_name)

        else: # Default to Ollama
            if not OLLAMA_AVAILABLE:
                raise HTTPException(status_code=503, detail="Ollama client is not available. Check connection to Ollama.")

            response = ollama_client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False
            )
            response_content = response['message']['content']
            return GenerationResponse(response=response_content, model_used=model_name)

    except Exception as e:
        # Log the exception details here in a real application
        print(f"❌ Error during model generation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the response: {str(e)}")

@app.get("/models/available", summary="List Available Models")
def get_available_models():
    """Returns a list of available models from all configured providers."""
    models = {"groq": [], "ollama": []}

    if groq_client:
        # These are common, high-performance models available on Groq.
        # A more dynamic approach would be to query an endpoint if Groq provided one.
        models["groq"] = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "gemma2-9b-it"
        ]

    if OLLAMA_AVAILABLE:
        try:
            installed_models = ollama_client.list()
            models["ollama"] = [model['name'] for model in installed_models['models']]
        except Exception as e:
            print(f"Could not fetch Ollama models: {e}")
            # Return empty list for ollama if it fails during the call
            models["ollama"] = []

    return models

@app.post("/embed", response_model=EmbedResponse, summary="Generate Embeddings for Texts")
async def generate_embeddings(request: EmbedRequest):
    """
    Receives a list of texts and returns their embeddings using the nomic-embed-text model.
    """
    texts = request.texts
    
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided for embedding.")
    
    if not OLLAMA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ollama client is not available. Check connection to Ollama.")
    
    try:
        embeddings = []
        model_name = "nomic-embed-text"
        
        for text in texts:
            # Generate embedding for each text
            response = ollama_client.embeddings(model=model_name, prompt=text)
            
            # Extract embedding from response
            if hasattr(response, 'embedding') and response.embedding:
                embeddings.append(response.embedding)
            elif isinstance(response, dict) and 'embedding' in response:
                embeddings.append(response['embedding'])
            else:
                # Handle unexpected response format
                raise Exception(f"Unexpected response format from Ollama: {type(response)}")
        
        return EmbedResponse(embeddings=embeddings, model_used=model_name)
    
    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating embeddings: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))  # Cloud Run için PORT, yerel için 8002
    uvicorn.run("main:app", host="0.0.0.0", port=port)
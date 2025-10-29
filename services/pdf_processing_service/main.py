import os
import tempfile
import logging
import traceback
from typing import Dict, Any, Tuple

# Set environment variables BEFORE any other imports
def _setup_marker_environment():
    """Setup marker environment variables before any marker imports."""
    cache_base = os.getenv("MARKER_CACHE_DIR", "/app/models")
    cache_vars = {
        "TORCH_HOME": f"{cache_base}/torch",
        "HUGGINGFACE_HUB_CACHE": f"{cache_base}/huggingface",
        "TRANSFORMERS_CACHE": f"{cache_base}/transformers",
        "HF_HOME": f"{cache_base}/hf_home",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "MARKER_CACHE_DIR": cache_base,
        "MARKER_DISABLE_GEMINI": "true",
        "MARKER_USE_LOCAL_ONLY": "true",
        "MARKER_DISABLE_ALL_LLM": "true",
    }
    for key, value in cache_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    logging.info(f"Marker environment configured. Cache base: {cache_base}")

_setup_marker_environment()

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import PyPDF2

# Dynamic import for Marker
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
    logging.info("Marker library loaded successfully.")
except ImportError:
    MARKER_AVAILABLE = False
    logging.warning("Marker library not found. Fallback to PyPDF2 will be used.")

# --- FastAPI App Definition ---
app = FastAPI(
    title="PDF Processing Service",
    description="A microservice to convert PDF files to high-quality Markdown using Marker.",
    version="1.0.0"
)

class PDFMetadata(BaseModel):
    source_file: str
    processing_method: str
    text_length: int
    page_count: int

class PDFProcessingResponse(BaseModel):
    content: str
    metadata: PDFMetadata

# --- Fallback PDF Processor ---
def fallback_pdf_extract(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extracts text from a PDF using PyPDF2 as a fallback."""
    logging.info(f"Using fallback PDF extraction for {pdf_path}")
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            metadata = {
                "source_file": os.path.basename(pdf_path),
                "processing_method": "pypdf2_fallback",
                "text_length": len(text),
                "page_count": len(reader.pages),
            }
            return text, metadata
    except Exception as e:
        logging.error(f"Fallback PDF extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fallback PDF processing failed: {e}")

# --- Marker PDF Processor ---
class MarkerProcessor:
    def __init__(self):
        self.converter = None
        self.models_loaded = False
        if MARKER_AVAILABLE:
            self._load_converter()

    def _load_converter(self):
        if self.models_loaded:
            return
        try:
            logging.info("Loading Marker models from cache...")
            artifact_dict = create_model_dict()
            self.converter = PdfConverter(artifact_dict=artifact_dict)
            self.models_loaded = True
            logging.info("Marker models loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Marker models: {e}\n{traceback.format_exc()}")
            self.models_loaded = False

    def process(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        if not self.models_loaded or not self.converter:
            logging.warning("Marker not available, using fallback.")
            return fallback_pdf_extract(pdf_path)

        try:
            logging.info(f"Processing with Marker: {pdf_path}")
            rendered = self.converter(pdf_path)
            markdown_text, _, _ = text_from_rendered(rendered)
            
            page_count = len(rendered.children) if hasattr(rendered, 'children') else 0

            metadata = {
                "source_file": os.path.basename(pdf_path),
                "processing_method": "marker",
                "text_length": len(markdown_text),
                "page_count": page_count,
            }
            return markdown_text, metadata
        except Exception as e:
            logging.error(f"Marker processing failed: {e}\n{traceback.format_exc()}")
            logging.warning("Falling back to PyPDF2 due to Marker error.")
            return fallback_pdf_extract(pdf_path)

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    logging.basicConfig(level=logging.INFO)
    if MARKER_AVAILABLE:
        # Pre-load models on startup
        app.state.processor = MarkerProcessor()
    else:
        app.state.processor = None
    logging.info("PDF Processing Service started.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "marker_available": MARKER_AVAILABLE}

@app.post("/process", response_model=PDFProcessingResponse)
async def process_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF is supported.")

    # Save uploaded file to a temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {e}")

    try:
        if app.state.processor:
            content, metadata = app.state.processor.process(tmp_path)
        else:
            content, metadata = fallback_pdf_extract(tmp_path)
        
        return PDFProcessingResponse(content=content, metadata=metadata)
    
    except HTTPException as e:
        # Re-raise HTTP exceptions from processors
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # Clean up the temporary file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
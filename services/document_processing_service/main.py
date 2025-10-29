import os
import uuid
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing Service",
    description="Metin işleme ve ChromaDB entegrasyonu için mikro servis",
    version="1.0.0"
)

# Pydantic modelleri
class ProcessRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}
    collection_name: Optional[str] = "documents"
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

class ProcessResponse(BaseModel):
    success: bool
    message: str
    chunks_processed: int
    collection_name: str
    chunk_ids: List[str]

# Ortam değişkenleri
MODEL_INFERENCER_URL = os.getenv("MODEL_INFERENCER_URL", "http://model-inferencer:8002")
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://chromadb:8000")
PORT = int(os.getenv("PORT", "8003"))

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = None
        self.chroma_client = None
        
    def initialize_text_splitter(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Text splitter'ı başlat"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def initialize_chroma_client(self):
        """ChromaDB client'ını başlat"""
        try:
            self.chroma_client = chromadb.HttpClient(host=CHROMADB_URL.replace("http://", "").split(":")[0], 
                                                   port=int(CHROMADB_URL.split(":")[-1]))
            logger.info(f"ChromaDB client başarıyla bağlandı: {CHROMADB_URL}")
        except Exception as e:
            logger.error(f"ChromaDB bağlantı hatası: {str(e)}")
            raise HTTPException(status_code=500, detail=f"ChromaDB bağlantı hatası: {str(e)}")
    
    def split_text(self, text: str) -> List[str]:
        """Metni chunk'lara ayır"""
        if not self.text_splitter:
            raise HTTPException(status_code=500, detail="Text splitter başlatılmamış")
        
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Metin {len(chunks)} chunk'a ayrıldı")
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Model inference servisinden embedding'leri al"""
        try:
            embed_url = f"{MODEL_INFERENCER_URL}/embed"
            
            embeddings = []
            for text in texts:
                response = requests.post(
                    embed_url,
                    json={"text": text},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"Embedding hatası: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Embedding alma hatası: {response.status_code}"
                    )
                
                embedding_data = response.json()
                embeddings.append(embedding_data.get("embedding", []))
            
            logger.info(f"{len(embeddings)} embedding başarıyla alındı")
            return embeddings
            
        except requests.RequestException as e:
            logger.error(f"Model inference servis isteği hatası: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Model inference servis hatası: {str(e)}"
            )
    
    def store_in_chromadb(self, chunks: List[str], embeddings: List[List[float]], 
                         metadata: Dict[str, Any], collection_name: str) -> List[str]:
        """Chunk'ları ve embedding'leri ChromaDB'ye kaydet"""
        if not self.chroma_client:
            raise HTTPException(status_code=500, detail="ChromaDB client başlatılmamış")
        
        try:
            # Collection'ı al veya oluştur
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
            except Exception:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Document chunks with embeddings"}
                )
            
            # Her chunk için benzersiz ID oluştur
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            
            # Her chunk için metadata hazırla
            chunk_metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "chunk_id": chunk_ids[i]
                })
                chunk_metadatas.append(chunk_metadata)
            
            # ChromaDB'ye ekle
            collection.add(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"{len(chunks)} chunk ChromaDB'ye kaydedildi. Collection: {collection_name}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"ChromaDB kaydetme hatası: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"ChromaDB kaydetme hatası: {str(e)}"
            )

# Global processor instance
processor = DocumentProcessor()

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında gerekli bağlantıları başlat"""
    logger.info("Document Processing Service başlatılıyor...")
    processor.initialize_chroma_client()
    logger.info("Service başarıyla başlatıldı")

@app.get("/")
async def root():
    """Health check endpoint'i"""
    return {"message": "Document Processing Service çalışıyor", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detaylı health check"""
    try:
        # ChromaDB bağlantısını test et
        if processor.chroma_client:
            processor.chroma_client.heartbeat()
        
        # Model inference servisini test et
        health_response = requests.get(f"{MODEL_INFERENCER_URL}/health", timeout=5)
        model_service_healthy = health_response.status_code == 200
        
        return {
            "status": "healthy",
            "chromadb_connected": processor.chroma_client is not None,
            "model_service_connected": model_service_healthy,
            "model_inferencer_url": MODEL_INFERENCER_URL,
            "chromadb_url": CHROMADB_URL
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "chromadb_connected": False,
            "model_service_connected": False
        }

@app.post("/process-and-store", response_model=ProcessResponse)
async def process_and_store(request: ProcessRequest):
    """
    Metin bloğunu işle ve ChromaDB'ye kaydet
    
    1. Metni chunk'lara ayırır
    2. Her chunk için embedding alır  
    3. ChromaDB'ye kaydeder
    """
    try:
        logger.info(f"Metin işleme başlatıldı. Uzunluk: {len(request.text)} karakter")
        
        # Text splitter'ı başlat
        processor.initialize_text_splitter(request.chunk_size, request.chunk_overlap)
        
        # Metni chunk'lara ayır
        chunks = processor.split_text(request.text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Metin chunk'lara ayrılamadı")
        
        # Embedding'leri al
        embeddings = processor.get_embeddings(chunks)
        
        if len(embeddings) != len(chunks):
            raise HTTPException(
                status_code=500, 
                detail="Embedding sayısı chunk sayısıyla eşleşmiyor"
            )
        
        # ChromaDB'ye kaydet
        chunk_ids = processor.store_in_chromadb(
            chunks=chunks,
            embeddings=embeddings,
            metadata=request.metadata,
            collection_name=request.collection_name
        )
        
        logger.info(f"İşlem tamamlandı. {len(chunks)} chunk işlendi.")
        
        return ProcessResponse(
            success=True,
            message=f"Başarıyla işlendi: {len(chunks)} chunk kaydedildi",
            chunks_processed=len(chunks),
            collection_name=request.collection_name,
            chunk_ids=chunk_ids
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"İşlem hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"İşlem hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
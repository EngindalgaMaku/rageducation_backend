import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from typing import Dict, Any

# Proje bileşenlerini import et
from src.config import get_config
from src.utils.helpers import setup_logging
from src.analytics.database import DatabaseAnalytics, get_experiment_db
from src.vector_store.faiss_store import FaissVectorStore
from src.query_processing.query_processor import QueryProcessor
from src.rag.rag_pipeline import RAGPipeline
from src.qa.qa_service import QAService
from src.services.learning_loop_manager import LearningLoopManager
from src.api import feedback_api
from pydantic import BaseModel

# Global değişkenler (lifespan içinde başlatılacak)
app_state: Dict[str, Any] = {}

# Logger'ı başlat
logger = setup_logging()

# Pydantic modelleri
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    session_id: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI uygulama yaşam döngüsü yöneticisi.
    Uygulama başlarken servisleri başlatır, kapanırken durdurur.
    """
    logger.info("Uygulama başlatılıyor... Servisler yükleniyor.")
    
    # 1. Yapılandırmayı yükle
    config = get_config()
    app_state["config"] = config
    
    # 2. Veritabanı bağlantısını kur
    db_conn = get_experiment_db()
    app_state["db_connection"] = db_conn
    
    # 3. Vektör deposunu yükle
    vector_store = FaissVectorStore(index_path=config["vector_store_path"])
    app_state["vector_store"] = vector_store
    
    # 4. Diğer servisleri başlat
    query_processor = QueryProcessor(config)
    rag_pipeline = RAGPipeline(config, vector_store)
    qa_service = QAService(config, query_processor, rag_pipeline, db_conn)
    app_state["qa_service"] = qa_service
    
    # 5. Öğrenme döngüsü yöneticisini başlat
    learning_loop_manager = LearningLoopManager(
        db_connection=db_conn,
        analysis_interval_seconds=config.get("analysis_interval_seconds", 3600) # 1 saat
    )
    learning_loop_manager.start()
    app_state["learning_loop_manager"] = learning_loop_manager
    
    logger.info("Tüm servisler başarıyla başlatıldı.")
    
    yield
    
    # Uygulama kapandığında çalışacak kod
    logger.info("Uygulama kapatılıyor... Servisler durduruluyor.")
    app_state["learning_loop_manager"].stop()
    logger.info("Öğrenme döngüsü yöneticisi durduruldu.")


# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Eğitimde RAG Destekli Bilgi Erişim Sistemi - API",
    description="Aktif öğrenme ve geri bildirim döngüsü için API endpoint'leri.",
    version="1.0.0",
    lifespan=lifespan
)

# API yönlendiricilerini dahil et
app.include_router(feedback_api.router, prefix="/api")

@app.get("/", tags=["Root"])
async def read_root():
    """
    API'nin çalıştığını doğrulayan kök endpoint.
    """
    return {"message": "API sunucusu çalışıyor."}

@app.post("/api/query", tags=["QA"])
async def handle_query(request: QueryRequest, qa_service: QAService = Depends(lambda: app_state.get("qa_service"))):
    """
    Kullanıcı sorgusunu işler ve bir cevap döndürür.
    """
    if not qa_service:
        raise HTTPException(status_code=503, detail="QA Servisi henüz hazır değil.")
    
    try:
        response = qa_service.answer_question(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id
        )
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        return response
    except Exception as e:
        logger.error(f"Sorgu işlenirken hata oluştu: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Sorgu işlenirken beklenmedik bir hata oluştu.")


if __name__ == "__main__":
    logger.info("API sunucusu başlatılıyor...")
    # Sunucuyu `uvicorn` ile çalıştır.
    # Terminalden çalıştırmak için: uvicorn src.api_server:app --reload
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=True)
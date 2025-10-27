#!/usr/bin/env python3
"""
Document loading script for RAG system.
This script processes all documents in the uploads directory and loads them into the vector store.
"""

import os
import glob
from src.app_logic import get_store, add_document_to_store
from src import config
from src.utils.helpers import setup_logging

logger = setup_logging()

def load_all_documents():
    """Load all documents from uploads directory into the main vector store."""
    
    # Get the main vector store
    vector_store = get_store(config.VECTOR_STORE_PATH)
    logger.info(f"Using vector store: {config.VECTOR_STORE_PATH}")
    
    # Check uploads directory
    uploads_dir = "data/uploads"
    if not os.path.exists(uploads_dir):
        logger.warning(f"Uploads directory {uploads_dir} does not exist")
        return
    
    # Supported file extensions
    extensions = ["*.pdf", "*.docx", "*.pptx"]
    
    total_processed = 0
    total_chunks = 0
    
    for ext in extensions:
        pattern = os.path.join(uploads_dir, ext)
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} {ext} files")
        
        for file_path in files:
            filename = os.path.basename(file_path)
            logger.info(f"Processing: {filename}")
            
            try:
                # Read file
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                
                # Process document with reduced chunk size for context length
                result = add_document_to_store(
                    file_bytes=file_bytes,
                    filename=filename,
                    vector_store=vector_store,
                    strategy="sentence",  # Use sentence strategy for better semantic chunking
                    chunk_size=800,       # Optimized chunk size for coherence
                    chunk_overlap=150,    # Better overlap for context
                    embedding_model=config.OLLAMA_EMBEDDING_MODEL
                )
                
                chunks_added = result.get("chunks", 0)
                documents_added = result.get("added", 0)
                
                logger.info(f"Processed {filename}: {documents_added} documents added from {chunks_added} chunks")
                total_processed += 1
                total_chunks += chunks_added
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
    
    logger.info(f"\n🎉 Processing complete!")
    logger.info(f"📊 Total files processed: {total_processed}")
    logger.info(f"📊 Total chunks created: {total_chunks}")
    
    # Check final vector store size
    if vector_store.index:
        total_docs = vector_store.index.ntotal
        logger.info(f"📊 Total documents in vector store: {total_docs}")
    else:
        logger.warning("No vector store index found")

if __name__ == "__main__":
    print("🚀 Starting document loading process...")
    load_all_documents()
    print("✨ Done!")
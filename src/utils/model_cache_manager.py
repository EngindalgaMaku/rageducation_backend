"""
Model Cache Manager for Marker PDF Processing
Handles persistent caching of large ML models to prevent repeated downloads
"""

import os
import logging
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import psutil
import threading
import time

logger = logging.getLogger(__name__)

class ModelCacheManager:
    """
    Centralized model cache manager for Marker PDF processing.
    Handles persistent storage and retrieval of ML models to prevent repeated downloads.
    """
    
    def __init__(self):
        self.base_cache_dir = Path(os.getenv("MARKER_CACHE_DIR", "/app/models"))
        self.models_dir = self.base_cache_dir / "marker_models"
        self.cache_info_file = self.models_dir / "cache_info.json"
        self.lock = threading.Lock()
        self._ensure_cache_directories()
        self._load_cache_info()
    
    def _ensure_cache_directories(self):
        """Ensure cache directories exist with proper permissions"""
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            # Set permissions for container environments
            if os.name != 'nt':  # Unix-like systems
                os.chmod(self.models_dir, 0o755)
                
            logger.info(f"📁 Model cache directory initialized: {self.models_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to create cache directory: {e}")
            # Fallback to temp directory
            import tempfile
            self.base_cache_dir = Path(tempfile.gettempdir()) / "marker_cache"
            self.models_dir = self.base_cache_dir / "marker_models"
            self.models_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"⚠️ Using fallback cache directory: {self.models_dir}")
    
    def _load_cache_info(self):
        """Load cache metadata information"""
        self.cache_info = {}
        try:
            if self.cache_info_file.exists():
                with open(self.cache_info_file, 'r') as f:
                    self.cache_info = json.load(f)
                logger.info(f"📋 Loaded cache info: {len(self.cache_info)} cached model sets")
        except Exception as e:
            logger.warning(f"⚠️ Could not load cache info: {e}")
            self.cache_info = {}
    
    def _save_cache_info(self):
        """Save cache metadata information"""
        try:
            with open(self.cache_info_file, 'w') as f:
                json.dump(self.cache_info, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"⚠️ Could not save cache info: {e}")
    
    def _get_model_cache_key(self, model_config: Dict[str, Any]) -> str:
        """Generate a unique cache key for model configuration"""
        # Create a hash based on model configuration
        config_str = json.dumps(model_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_cached_model_dict(self, force_download: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get cached model dictionary or download if not available
        
        Args:
            force_download: Force fresh download even if cache exists
            
        Returns:
            Model dictionary or None if failed
        """
        with self.lock:
            cache_key = "default_marker_models"
            cached_path = self.models_dir / cache_key
            
            # Check if models are already cached and valid
            if not force_download and cached_path.exists() and cache_key in self.cache_info:
                try:
                    logger.info(f"✅ Using cached models from: {cached_path}")
                    return self._load_cached_models(cached_path)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load cached models: {e}")
                    # Continue to download fresh models
            
            # Download and cache models
            return self._download_and_cache_models(cache_key, cached_path)
    
    def _load_cached_models(self, cached_path: Path) -> Dict[str, Any]:
        """Load models from cache"""
        # CRITICAL: Environment variables should already be set at process start
        # Just verify they point to the right location
        expected_torch = str(cached_path / "torch")
        actual_torch = os.environ.get("TORCH_HOME", "")
        
        if actual_torch != expected_torch:
            logger.warning(f"⚠️ TORCH_HOME mismatch: expected {expected_torch}, got {actual_torch}")
            # Don't override - let the process-level env vars take precedence
        
        memory_before = self._get_memory_usage()
        logger.info(f"🔄 Loading cached models from environment... (Memory: {memory_before:.1f}MB)")
        logger.info(f"🔧 TORCH_HOME: {os.environ.get('TORCH_HOME')}")
        logger.info(f"🔧 TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
        
        # Verify cache exists
        torch_cache = os.environ.get("TORCH_HOME", "")
        if not os.path.exists(torch_cache):
            logger.error(f"❌ Cache directory missing: {torch_cache}")
            raise FileNotFoundError(f"Model cache not found: {torch_cache}")
        
        # Import here to use cached models
        from marker.models import create_model_dict
        
        # Load models with cache paths already set in environment
        model_dict = create_model_dict()
        
        memory_after = self._get_memory_usage()
        logger.info(f"✅ Environment-cached models loaded! (Memory: {memory_after:.1f}MB, +{memory_after-memory_before:.1f}MB)")
        
        return model_dict
    
    def _download_and_cache_models(self, cache_key: str, cached_path: Path) -> Optional[Dict[str, Any]]:
        """Download fresh models and cache them"""
        try:
            logger.info(f"📥 Downloading and caching Marker models...")
            memory_before = self._get_memory_usage()
            
            # Create cache subdirectories
            torch_cache = cached_path / "torch"
            hf_cache = cached_path / "huggingface"
            transformers_cache = cached_path / "transformers"
            hf_home = cached_path / "hf_home"
            
            for cache_dir in [torch_cache, hf_cache, transformers_cache, hf_home]:
                cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Set environment variables for model download locations
            os.environ["TORCH_HOME"] = str(torch_cache)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
            os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
            os.environ["HF_HOME"] = str(hf_home)
            
            # Additional marker-specific cache settings
            os.environ["MARKER_CACHE_DIR"] = str(cached_path)
            os.environ["MARKER_MODELS_DIR"] = str(cached_path / "marker_models")
            
            logger.info(f"🏗️ Model cache directories prepared at: {cached_path}")
            
            # Import and download models
            from marker.models import create_model_dict
            
            start_time = time.time()
            model_dict = create_model_dict()
            download_time = time.time() - start_time
            
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            # Update cache info
            self.cache_info[cache_key] = {
                "created_at": datetime.now().isoformat(),
                "cache_path": str(cached_path),
                "download_time_seconds": download_time,
                "memory_used_mb": memory_used,
                "model_count": len(model_dict) if model_dict else 0
            }
            self._save_cache_info()
            
            logger.info(f"✅ Models cached successfully!")
            logger.info(f"📊 Download time: {download_time:.1f}s, Memory used: {memory_used:.1f}MB")
            logger.info(f"📁 Cache location: {cached_path}")
            
            return model_dict
            
        except Exception as e:
            logger.error(f"❌ Failed to download and cache models: {e}")
            logger.error(f"📋 Error details: {str(e)}")
            
            # Clean up partial cache
            if cached_path.exists():
                try:
                    shutil.rmtree(cached_path)
                except:
                    pass
            
            return None
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cached models"""
        with self.lock:
            if cache_key:
                cached_path = self.models_dir / cache_key
                if cached_path.exists():
                    shutil.rmtree(cached_path)
                    if cache_key in self.cache_info:
                        del self.cache_info[cache_key]
                        self._save_cache_info()
                    logger.info(f"🗑️ Cleared cache for: {cache_key}")
            else:
                # Clear all caches
                if self.models_dir.exists():
                    shutil.rmtree(self.models_dir)
                    self._ensure_cache_directories()
                self.cache_info = {}
                self._save_cache_info()
                logger.info(f"🗑️ Cleared all model caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = 0
        if self.models_dir.exists():
            for path in self.models_dir.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        
        return {
            "cache_directory": str(self.models_dir),
            "total_cache_size_mb": total_size / (1024 * 1024),
            "cached_model_sets": len(self.cache_info),
            "cache_info": self.cache_info,
            "disk_usage": {
                "total_gb": shutil.disk_usage(self.models_dir).total / (1024**3),
                "used_gb": shutil.disk_usage(self.models_dir).used / (1024**3),
                "free_gb": shutil.disk_usage(self.models_dir).free / (1024**3)
            }
        }

# Global instance
_cache_manager = None

def get_model_cache_manager() -> ModelCacheManager:
    """Get the global model cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ModelCacheManager()
    return _cache_manager

def get_cached_marker_models(force_download: bool = False) -> Optional[Dict[str, Any]]:
    """
    Get cached Marker models with automatic download if needed
    
    Args:
        force_download: Force fresh download even if cache exists
        
    Returns:
        Model dictionary or None if failed
    """
    return get_model_cache_manager().get_cached_model_dict(force_download)

if __name__ == "__main__":
    # Test the cache manager
    cache_manager = get_model_cache_manager()
    print("📋 Cache Manager Test")
    
    # Get cache stats
    stats = cache_manager.get_cache_stats()
    print(f"📊 Cache Stats: {stats}")
    
    # Test model loading
    print("🔄 Testing model cache...")
    models = get_cached_marker_models()
    
    if models:
        print(f"✅ Models loaded successfully! Count: {len(models)}")
    else:
        print("❌ Failed to load models")
    
    # Final stats
    final_stats = cache_manager.get_cache_stats()
    print(f"📊 Final Stats: {final_stats}")
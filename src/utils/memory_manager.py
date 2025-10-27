"""
Memory management utilities for improved performance and stability.
"""

import gc
import psutil
import weakref
from typing import Any, Optional
from ..utils.logger import get_logger

class MemoryManager:
    """
    Memory management utility class.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__, config)
        self._object_cache = weakref.WeakValueDictionary()
        
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "total_mb": psutil.virtual_memory().total / 1024 / 1024
        }
    
    def should_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        memory_usage = self.get_memory_usage()
        
        # Trigger GC if memory usage is high
        if memory_usage["percent"] > 80:
            return True
            
        # Trigger GC if available memory is low
        if memory_usage["available_mb"] < 1024:  # Less than 1GB available
            return True
            
        return False
    
    def force_gc(self) -> dict:
        """Force garbage collection and return statistics."""
        self.logger.info("Forcing garbage collection...")
        
        memory_before = self.get_memory_usage()
        
        # Run garbage collection
        collected = gc.collect()
        
        memory_after = self.get_memory_usage()
        
        freed_mb = memory_before["rss_mb"] - memory_after["rss_mb"]
        
        stats = {
            "objects_collected": collected,
            "memory_freed_mb": freed_mb,
            "memory_before": memory_before,
            "memory_after": memory_after
        }
        
        self.logger.info(f"GC completed: collected {collected} objects, freed {freed_mb:.2f}MB")
        
        return stats
    
    def auto_gc_check(self) -> Optional[dict]:
        """Automatically check and perform GC if needed."""
        if self.should_gc():
            return self.force_gc()
        return None
    
    def cache_object(self, key: str, obj: Any) -> None:
        """Cache an object using weak references."""
        self._object_cache[key] = obj
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """Get a cached object."""
        return self._object_cache.get(key)
    
    def clear_cache(self) -> int:
        """Clear the object cache."""
        count = len(self._object_cache)
        self._object_cache.clear()
        self.logger.info(f"Cleared {count} cached objects")
        return count
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_objects": len(self._object_cache),
            "cache_keys": list(self._object_cache.keys())
        }

# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None

def get_memory_manager(config: dict) -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(config)
    return _memory_manager

def cleanup_memory(config: dict) -> dict:
    """Utility function for memory cleanup."""
    manager = get_memory_manager(config)
    
    # Clear caches
    cache_cleared = manager.clear_cache()
    
    # Force garbage collection
    gc_stats = manager.force_gc()
    
    return {
        "cache_cleared": cache_cleared,
        "gc_stats": gc_stats
    }
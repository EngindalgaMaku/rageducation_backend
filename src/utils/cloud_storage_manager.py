"""
Cloud Storage Manager for RAG3
Handles persistent database storage for stateless Cloud Run deployments
"""

import os
import sqlite3
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager
import threading

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

from src.config import config

class CloudStorageManager:
    """
    Manages persistent database storage for cloud deployments.
    Syncs SQLite database with Google Cloud Storage to handle Cloud Run restarts.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.is_cloud = config.is_cloud_environment()
        
        # Local paths
        self.local_db_dir = Path('/tmp/rag3/databases')
        self.local_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Cloud storage client
        self._storage_client = None
        self._bucket = None
        
        # Thread-safe database operations
        self._db_lock = threading.RLock()
        
        # Initialize cloud storage if needed
        if self.is_cloud:
            self._init_cloud_storage()
    
    def _init_cloud_storage(self):
        """Initialize Google Cloud Storage client"""
        if not GCS_AVAILABLE:
            self.logger.error("Google Cloud Storage not available - install google-cloud-storage")
            return
        
        try:
            self._storage_client = storage.Client()
            bucket_name = config.database_config.get('storage_bucket', 'rag3-data')
            self._bucket = self._storage_client.bucket(bucket_name)
            
            # Test bucket access
            if not self._bucket.exists():
                self.logger.warning(f"GCS bucket '{bucket_name}' does not exist - creating it")
                self._bucket = self._storage_client.create_bucket(bucket_name)
            
            self.logger.info(f"Cloud storage initialized: {bucket_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cloud storage: {e}")
            self._storage_client = None
            self._bucket = None
    
    def get_database_path(self, db_name: str = "sessions.db") -> str:
        """
        Get the local database path, downloading from cloud storage if needed.
        
        Args:
            db_name: Name of the database file
            
        Returns:
            str: Local path to database file
        """
        local_db_path = self.local_db_dir / db_name
        
        if self.is_cloud and self._bucket:
            # Try to download latest database from cloud storage
            self._download_database(db_name, str(local_db_path))
        
        return str(local_db_path)
    
    def _download_database(self, db_name: str, local_path: str) -> bool:
        """Download database from cloud storage"""
        if not self._bucket:
            return False
        
        try:
            storage_path = config.database_config.get('storage_path', 'databases/') + db_name
            blob = self._bucket.blob(storage_path)
            
            if blob.exists():
                self.logger.info(f"Downloading database from GCS: {storage_path}")
                blob.download_to_filename(local_path)
                self.logger.info(f"Database downloaded to: {local_path}")
                return True
            else:
                self.logger.info(f"No existing database in GCS: {storage_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to download database: {e}")
            return False
    
    def sync_database(self, db_name: str = "sessions.db") -> bool:
        """
        Sync local database to cloud storage.
        Called after database modifications.
        
        Args:
            db_name: Name of the database file
            
        Returns:
            bool: Success status
        """
        if not self.is_cloud or not self._bucket:
            return True  # No sync needed for local development
        
        local_db_path = self.local_db_dir / db_name
        
        if not local_db_path.exists():
            self.logger.warning(f"Database file does not exist: {local_db_path}")
            return False
        
        try:
            with self._db_lock:
                # Create a backup copy for upload
                temp_path = local_db_path.with_suffix('.tmp')
                shutil.copy2(local_db_path, temp_path)
                
                try:
                    storage_path = config.database_config.get('storage_path', 'databases/') + db_name
                    blob = self._bucket.blob(storage_path)
                    
                    # Upload with metadata
                    blob.metadata = {
                        'uploaded_at': str(int(time.time())),
                        'source': 'rag3-api',
                        'version': '1.0'
                    }
                    
                    blob.upload_from_filename(str(temp_path))
                    self.logger.info(f"Database synced to GCS: {storage_path}")
                    return True
                    
                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
                        
        except Exception as e:
            self.logger.error(f"Failed to sync database: {e}")
            return False
    
    @contextmanager
    def get_database_connection(self, db_name: str = "sessions.db", timeout: float = 30.0):
        """
        Context manager for database connections with automatic sync.
        
        Args:
            db_name: Name of the database file
            timeout: Connection timeout in seconds
            
        Yields:
            sqlite3.Connection: Database connection
        """
        # Get database path (downloads from cloud if needed)
        db_path = self.get_database_path(db_name)
        
        with self._db_lock:
            conn = sqlite3.connect(db_path, timeout=timeout)
            conn.row_factory = sqlite3.Row
            
            try:
                yield conn
                conn.commit()
                
                # Sync to cloud after successful transaction
                if conn.in_transaction is False:  # Only sync if transaction completed
                    self.sync_database(db_name)
                    
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Database operation failed: {e}")
                raise
            finally:
                conn.close()
    
    def backup_database(self, db_name: str = "sessions.db", backup_name: Optional[str] = None) -> Optional[str]:
        """
        Create a backup of the database in cloud storage.
        
        Args:
            db_name: Name of the database file
            backup_name: Custom backup name (auto-generated if None)
            
        Returns:
            str: Backup path in cloud storage, or None if failed
        """
        if not self.is_cloud or not self._bucket:
            return None
        
        local_db_path = self.local_db_dir / db_name
        if not local_db_path.exists():
            return None
        
        try:
            if not backup_name:
                timestamp = int(time.time())
                backup_name = f"{db_name}.backup.{timestamp}"
            
            backup_path = f"backups/databases/{backup_name}"
            blob = self._bucket.blob(backup_path)
            
            blob.metadata = {
                'backup_created_at': str(int(time.time())),
                'original_db': db_name,
                'backup_type': 'manual'
            }
            
            blob.upload_from_filename(str(local_db_path))
            self.logger.info(f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create database backup: {e}")
            return None
    
    def restore_database(self, backup_path: str, db_name: str = "sessions.db") -> bool:
        """
        Restore database from a backup in cloud storage.
        
        Args:
            backup_path: Path to backup file in cloud storage
            db_name: Target database name
            
        Returns:
            bool: Success status
        """
        if not self.is_cloud or not self._bucket:
            return False
        
        try:
            blob = self._bucket.blob(backup_path)
            if not blob.exists():
                self.logger.error(f"Backup not found: {backup_path}")
                return False
            
            local_db_path = self.local_db_dir / db_name
            
            with self._db_lock:
                # Download backup to local path
                blob.download_to_filename(str(local_db_path))
                
                # Sync the restored database back to current location
                self.sync_database(db_name)
            
            self.logger.info(f"Database restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            return False
    
    def list_backups(self, db_name: str = "sessions.db") -> list:
        """List available database backups"""
        if not self.is_cloud or not self._bucket:
            return []
        
        try:
            prefix = f"backups/databases/{db_name}.backup."
            blobs = self._bucket.list_blobs(prefix=prefix)
            
            backups = []
            for blob in blobs:
                backups.append({
                    'name': blob.name,
                    'created': blob.time_created,
                    'size': blob.size,
                    'metadata': blob.metadata or {}
                })
            
            return sorted(backups, key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        stats = {
            'is_cloud': self.is_cloud,
            'gcs_available': GCS_AVAILABLE and self._bucket is not None,
            'local_db_dir': str(self.local_db_dir),
            'local_databases': []
        }
        
        # List local databases
        if self.local_db_dir.exists():
            for db_file in self.local_db_dir.glob('*.db'):
                stats['local_databases'].append({
                    'name': db_file.name,
                    'size': db_file.stat().st_size,
                    'modified': db_file.stat().st_mtime
                })
        
        # Add cloud storage stats if available
        if self.is_cloud and self._bucket:
            try:
                bucket_info = {
                    'name': self._bucket.name,
                    'location': self._bucket.location,
                    'storage_class': self._bucket.storage_class
                }
                stats['cloud_storage'] = bucket_info
            except Exception as e:
                stats['cloud_storage_error'] = str(e)
        
        return stats

# Global instance
cloud_storage_manager = CloudStorageManager()
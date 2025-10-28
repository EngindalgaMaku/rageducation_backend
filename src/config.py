"""
RAG3 Configuration Management
Cloud-Ready Database Configuration with Fallback Support
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class RAGConfig:
    """Centralized configuration management with cloud support"""
    
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.is_cloud = self.environment == 'production'
        self.logger = logging.getLogger(__name__)
        
        # Database Configuration
        self._setup_database_config()
        
        # Storage Configuration
        self._setup_storage_config()
        
        # Model Configuration
        self._setup_model_config()
        
    def _setup_database_config(self):
        """Setup database configuration with cloud support"""
        
        if self.is_cloud:
            # Cloud SQL or external database for production
            self.database_config = {
                'type': os.getenv('DB_TYPE', 'sqlite'),  # 'postgresql', 'sqlite'
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT', '5432'),
                'name': os.getenv('DB_NAME', 'rag3_sessions'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'ssl': os.getenv('DB_SSL', 'true') == 'true',
                # Cloud SQL connection for Google Cloud Run
                'cloud_sql_connection': os.getenv('CLOUD_SQL_CONNECTION_NAME'),
                # Fallback to Cloud Storage-backed SQLite
                'fallback_storage': 'gcs',  # 'gcs', 'local'
                'storage_bucket': os.getenv('GCS_BUCKET_NAME', 'rag3-data'),
                'storage_path': 'databases/'
            }
        else:
            # Local SQLite for development
            self.database_config = {
                'type': 'sqlite',
                'path': 'data/analytics/sessions.db',
                'fallback_storage': 'local'
            }
    
    def _setup_storage_config(self):
        """Setup file storage configuration"""
        
        if self.is_cloud:
            self.storage_config = {
                'type': 'gcs',  # Google Cloud Storage
                'bucket': os.getenv('GCS_BUCKET_NAME', 'rag3-data'),
                'vector_path': 'vector_stores/',
                'backup_path': 'backups/',
                'export_path': 'exports/',
                'markdown_path': 'markdown/',
                'temp_path': '/tmp/rag3/',  # Cloud Run temp space
            }
        else:
            self.storage_config = {
                'type': 'local',
                'vector_path': 'data/vector_db/sessions/',
                'backup_path': 'data/backups/sessions/',
                'export_path': 'data/exports/sessions/',
                'markdown_path': 'data/markdown/',
                'temp_path': 'data/temp/'
            }
    
    def _setup_model_config(self):
        """Setup model and caching configuration"""
        
        self.model_config = {
            'embedding_provider': os.getenv('EMBEDDING_PROVIDER', 'sentence_transformers'),
            'llm_provider': os.getenv('LLM_PROVIDER', 'groq'),
            'cache_models': os.getenv('CACHE_MODELS', 'true') == 'true',
            'model_cache_path': '/app/models' if self.is_cloud else 'models/',
            'max_memory_mb': int(os.getenv('MARKER_MAX_MEMORY_MB', '3500')),
            'timeout_seconds': int(os.getenv('MARKER_TIMEOUT_SECONDS', '900'))
        }
    
    def get_database_url(self) -> str:
        """Get database connection URL/path"""
        
        if self.database_config['type'] == 'postgresql':
            if self.database_config.get('cloud_sql_connection'):
                # Cloud SQL via Unix socket
                return (f"postgresql://{self.database_config['user']}:"
                       f"{self.database_config['password']}@/"
                       f"{self.database_config['name']}"
                       f"?host=/cloudsql/{self.database_config['cloud_sql_connection']}")
            else:
                # Standard PostgreSQL connection
                return (f"postgresql://{self.database_config['user']}:"
                       f"{self.database_config['password']}@"
                       f"{self.database_config['host']}:"
                       f"{self.database_config['port']}/"
                       f"{self.database_config['name']}")
        else:
            # SQLite - use cloud storage path if in production
            if self.is_cloud and self.database_config['fallback_storage'] == 'gcs':
                # This will be handled by the storage manager
                return 'cloud_storage'  
            else:
                return self.database_config.get('path', 'data/analytics/sessions.db')
    
    def get_storage_path(self, path_type: str) -> str:
        """Get storage path for different resource types"""
        return self.storage_config.get(f'{path_type}_path', '')
    
    def is_cloud_environment(self) -> bool:
        """Check if running in cloud environment"""
        return self.is_cloud
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            'environment': self.environment,
            'is_cloud': self.is_cloud,
            'database_type': self.database_config['type'],
            'storage_type': self.storage_config['type'],
            'embedding_provider': self.model_config['embedding_provider'],
            'llm_provider': self.model_config['llm_provider']
        }

# Global configuration instance
config = RAGConfig()

# Helper functions for backward compatibility
def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return config.database_config

def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration"""
    return config.storage_config

def is_cloud_environment() -> bool:
    """Check if running in cloud environment"""
    return config.is_cloud_environment()

def get_database_url() -> str:
    """Get database URL/path"""
    return config.get_database_url()

def setup_directories():
    """Setup necessary directories for local development"""
    if not config.is_cloud:
        directories = [
            'data/analytics',
            'data/vector_db/sessions',
            'data/backups/sessions',
            'data/exports/sessions',
            'data/markdown',
            'data/temp'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize directories on import
setup_directories()

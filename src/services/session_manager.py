"""
Professional Session Management System
Profesyonel Oturum Yönetim Sistemi

Bu modül öğretmenlerin ders oturumlarını veritabanı tabanlı olarak yönetmesine olanak sağlar:
- Oturum oluşturma, metadata ve validasyon ile
- Profesyonel kaydetme/export işlevleri  
- Güvenli silme, onay ve backup ile
- Oturum kategorizasyonu ve arama
- Import/Export yetenekleri
- Oturum analitikleri ve kullanım istatistikleri
"""

import os
import json
import sqlite3
import shutil
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
from contextlib import contextmanager

import logging
# from src.utils.logger import get_logger


class SessionStatus(Enum):
    """Oturum durumu"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DRAFT = "draft"
    COMPLETED = "completed"
    SUSPENDED = "suspended"


class SessionCategory(Enum):
    """Oturum kategorileri"""
    GENERAL = "general"
    SCIENCE = "science"
    MATHEMATICS = "mathematics"
    LANGUAGE = "language"
    SOCIAL_STUDIES = "social_studies"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    COMPUTER_SCIENCE = "computer_science"
    ART = "art"
    MUSIC = "music"
    PHYSICAL_EDUCATION = "physical_education"
    RESEARCH = "research"
    EXAM_PREP = "exam_prep"


@dataclass
class SessionMetadata:
    """Oturum metadata bilgileri"""
    session_id: str
    name: str
    description: str
    category: SessionCategory
    status: SessionStatus
    created_by: str
    created_at: str
    updated_at: str
    last_accessed: str
    grade_level: str
    subject_area: str
    learning_objectives: List[str]
    tags: List[str]
    document_count: int = 0
    total_chunks: int = 0
    query_count: int = 0
    avg_response_time: float = 0.0
    user_rating: float = 0.0
    notes: str = ""
    is_public: bool = False
    collaborators: List[str] = None
    backup_count: int = 0
    
    def __post_init__(self):
        if self.collaborators is None:
            self.collaborators = []


@dataclass
class SessionBackup:
    """Oturum yedekleme bilgileri"""
    backup_id: str
    session_id: str
    backup_path: str
    created_at: str
    backup_size: int
    description: str
    auto_created: bool = False


@dataclass 
class SessionExportData:
    """Oturum export verisi"""
    metadata: SessionMetadata
    vector_data: Dict[str, Any]
    performance_stats: Dict[str, Any]
    export_timestamp: str
    export_version: str = "1.0"


class ProfessionalSessionManager:
    """Profesyonel oturum yönetim sistemi"""
    
    def __init__(self, db_path: str = "data/analytics/sessions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Backup ve export dizinleri
        self.backup_dir = Path("data/backups/sessions")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.export_dir = Path("data/exports/sessions")  
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Database initialization
        self._init_schema()
        
        # Auto-cleanup old backups (30+ days)
        self._cleanup_old_backups()
    
    @contextmanager
    def get_connection(self):
        """Database connection context manager"""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            conn.close()
    
    def _init_schema(self):
        """Veritabanı şemasını oluştur"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_by TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    grade_level TEXT,
                    subject_area TEXT,
                    learning_objectives TEXT,  -- JSON array
                    tags TEXT,                 -- JSON array
                    document_count INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    query_count INTEGER DEFAULT 0,
                    avg_response_time REAL DEFAULT 0.0,
                    user_rating REAL DEFAULT 0.0,
                    notes TEXT,
                    is_public BOOLEAN DEFAULT 0,
                    collaborators TEXT,        -- JSON array
                    backup_count INTEGER DEFAULT 0
                )
            """)
            
            # Session backups table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_backups (
                    backup_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    backup_path TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    backup_size INTEGER,
                    description TEXT,
                    auto_created BOOLEAN DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Session activity log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_activity (
                    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    activity_type TEXT NOT NULL,  -- 'created', 'accessed', 'modified', 'backup', 'export'
                    description TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Session shares/collaborations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_shares (
                    share_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    shared_with TEXT NOT NULL,
                    permission_level TEXT DEFAULT 'read',  -- 'read', 'write', 'admin'
                    shared_by TEXT NOT NULL,
                    shared_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Document chunks table - stores text content of each generated chunk
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    chunk_metadata TEXT,  -- JSON format for additional chunk metadata
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_by ON sessions (created_by)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_category ON sessions (category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions (updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_activity_session_id ON session_activity (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_activity_timestamp ON session_activity (timestamp)")

            # Changelog table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS changelog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    date TEXT NOT NULL,
                    changes TEXT NOT NULL -- JSON array of strings
                )
            """)
            
            self.logger.info("Professional session management database schema initialized")
    
    def create_session(self, name: str, description: str, category: SessionCategory,
                      created_by: str, grade_level: str = "", subject_area: str = "",
                      learning_objectives: List[str] = None, tags: List[str] = None,
                      is_public: bool = False) -> SessionMetadata:
        """
        Profesyonel oturum oluştur - validasyon ve metadata ile
        
        Args:
            name: Oturum adı
            description: Açıklama  
            category: Kategori
            created_by: Oluşturan kişi
            grade_level: Sınıf seviyesi
            subject_area: Konu alanı
            learning_objectives: Öğrenme hedefleri
            tags: Etiketler
            is_public: Herkese açık mı
            
        Returns:
            SessionMetadata: Oluşturulan oturum metadatası
        """
        # Validation
        if not name.strip():
            raise ValueError("Session name cannot be empty")
        
        if len(name) > 100:
            raise ValueError("Session name too long (max 100 characters)")
        
        if self._session_exists(name, created_by):
            raise ValueError(f"Session '{name}' already exists for user {created_by}")
        
        # Generate unique session ID
        session_id = self._generate_session_id(name, created_by)
        
        # Create metadata
        now = datetime.now().isoformat()
        metadata = SessionMetadata(
            session_id=session_id,
            name=name.strip(),
            description=description.strip(),
            category=category,
            status=SessionStatus.DRAFT,  # Start as draft
            created_by=created_by,
            created_at=now,
            updated_at=now,
            last_accessed=now,
            grade_level=grade_level,
            subject_area=subject_area,
            learning_objectives=learning_objectives or [],
            tags=tags or [],
            is_public=is_public
        )
        
        # Save to database
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (
                    session_id, name, description, category, status, created_by,
                    created_at, updated_at, last_accessed, grade_level, subject_area,
                    learning_objectives, tags, is_public, collaborators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, metadata.name, metadata.description, category.value,
                metadata.status.value, created_by, now, now, now,
                grade_level, subject_area,
                json.dumps(learning_objectives or []),
                json.dumps(tags or []),
                is_public, json.dumps([])
            ))
        
        # Log activity
        self._log_activity(session_id, "created", f"Session '{name}' created", created_by)
        
        self.logger.info(f"Created professional session '{name}' with ID {session_id}")
        return metadata
    
    def get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Oturum metadatasını getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            return self._row_to_metadata(row)
    
    def update_session_metadata(self, session_id: str, **updates) -> bool:
        """Oturum metadatasını güncelle"""
        if not self._session_exists_by_id(session_id):
            return False
        
        allowed_fields = {
            'name', 'description', 'category', 'status', 'grade_level',
            'subject_area', 'learning_objectives', 'tags', 'notes', 
            'is_public', 'user_rating'
        }
        
        update_fields = []
        update_values = []
        
        for field, value in updates.items():
            if field in allowed_fields:
                if field in ['learning_objectives', 'tags']:
                    value = json.dumps(value if value else [])
                elif field == 'category' and hasattr(value, 'value'):
                    value = value.value
                elif field == 'status' and hasattr(value, 'value'):
                    value = value.value
                
                update_fields.append(f"{field} = ?")
                update_values.append(value)
        
        if not update_fields:
            return False
        
        update_fields.append("updated_at = ?")
        update_values.append(datetime.now().isoformat())
        update_values.append(session_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = f"UPDATE sessions SET {', '.join(update_fields)} WHERE session_id = ?"
            cursor.execute(query, update_values)
        
        # Log activity
        self._log_activity(session_id, "modified", f"Session metadata updated: {', '.join(updates.keys())}")
        
        return True
    
    def list_sessions(self, created_by: Optional[str] = None, 
                     category: Optional[SessionCategory] = None,
                     status: Optional[SessionStatus] = None,
                     limit: int = 50) -> List[SessionMetadata]:
        """Oturumları listele"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM sessions WHERE 1=1"
            params = []
            
            if created_by:
                query += " AND created_by = ?"
                params.append(created_by)
            
            if category:
                query += " AND category = ?"
                params.append(category.value)
                
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_metadata(row) for row in rows]
    
    def search_sessions(self, query: str, created_by: Optional[str] = None) -> List[SessionMetadata]:
        """Oturumlarda arama yap"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            search_query = """
                SELECT * FROM sessions 
                WHERE (name LIKE ? OR description LIKE ? OR tags LIKE ? OR notes LIKE ?)
            """
            params = [f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"]
            
            if created_by:
                search_query += " AND created_by = ?"
                params.append(created_by)
            
            search_query += " ORDER BY updated_at DESC LIMIT 20"
            
            cursor.execute(search_query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_metadata(row) for row in rows]
    
    def create_backup(self, session_id: str, description: str = "", 
                     auto_created: bool = False) -> SessionBackup:
        """Oturum yedeği oluştur"""
        metadata = self.get_session_metadata(session_id)
        if not metadata:
            raise ValueError(f"Session {session_id} not found")
        
        # Backup file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{session_id}_{timestamp}.zip"
        backup_path = self.backup_dir / backup_filename
        
        # Create backup ZIP
        vector_base_path = Path(f"data/vector_db/sessions/{session_id}")
        files_to_backup = [
            f"{vector_base_path}.index",
            f"{vector_base_path}.chunks", 
            f"{vector_base_path}.meta.jsonl"
        ]
        
        total_size = 0
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add session metadata (convert enums to serializable format)
            metadata_dict = self._metadata_to_dict(metadata)
            zipf.writestr("session_metadata.json", json.dumps(metadata_dict, indent=2))
            
            # Add vector files if they exist
            for file_path in files_to_backup:
                if Path(file_path).exists():
                    zipf.write(file_path, Path(file_path).name)
                    total_size += Path(file_path).stat().st_size
        
        # Create backup record
        backup_id = hashlib.md5(f"{session_id}_{timestamp}".encode()).hexdigest()
        backup = SessionBackup(
            backup_id=backup_id,
            session_id=session_id,
            backup_path=str(backup_path),
            created_at=datetime.now().isoformat(),
            backup_size=total_size,
            description=description or f"Backup of '{metadata.name}'",
            auto_created=auto_created
        )
        
        # Save backup record to database
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_backups 
                (backup_id, session_id, backup_path, created_at, backup_size, description, auto_created)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                backup_id, session_id, str(backup_path), backup.created_at,
                backup.backup_size, backup.description, auto_created
            ))
            
            # Update backup count
            cursor.execute(
                "UPDATE sessions SET backup_count = backup_count + 1 WHERE session_id = ?",
                (session_id,)
            )
        
        # Log activity
        self._log_activity(
            session_id, "backup",
            f"Backup created: {backup.description}", 
            metadata={'backup_id': backup_id, 'size': total_size}
        )
        
        self.logger.info(f"Created backup for session {session_id}: {backup_path}")
        return backup
    
    def delete_session(self, session_id: str, create_backup: bool = True,
                      deleted_by: Optional[str] = None) -> bool:
        """
        Oturumu güvenli şekilde sil
        
        Args:
            session_id: Silinecek oturum ID'si
            create_backup: Silmeden önce yedek oluştur
            deleted_by: Silen kişi
            
        Returns:
            bool: Silme işlemi başarılı mı
        """
        metadata = self.get_session_metadata(session_id)
        if not metadata:
            return False
        
        try:
            # Create backup before deletion if requested
            if create_backup:
                self.create_backup(
                    session_id, 
                    f"Pre-deletion backup of '{metadata.name}'",
                    auto_created=True
                )
            
            # Delete vector store files
            vector_base_path = Path(f"data/vector_db/sessions/{session_id}")
            files_to_delete = [
                f"{vector_base_path}.index",
                f"{vector_base_path}.chunks",
                f"{vector_base_path}.meta.jsonl"
            ]
            
            for file_path in files_to_delete:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            
            # Delete from database (cascading will handle related records)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            # Log activity (before deletion for record keeping)
            self._log_activity(
                session_id, "deleted",
                f"Session '{metadata.name}' deleted",
                user_id=deleted_by,
                metadata={'backup_created': create_backup}
            )
            
            self.logger.info(f"Deleted session {session_id} ('{metadata.name}')")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def export_session(self, session_id: str, export_format: str = "json") -> str:
        """
        Oturumu export et
        
        Args:
            session_id: Export edilecek oturum ID'si  
            export_format: Export formatı ('json', 'zip')
            
        Returns:
            str: Export dosyası yolu
        """
        metadata = self.get_session_metadata(session_id)
        if not metadata:
            raise ValueError(f"Session {session_id} not found")
        
        # Get performance stats
        performance_stats = self._get_session_performance_stats(session_id)
        
        # Prepare export data
        export_data = SessionExportData(
            metadata=metadata,
            vector_data=self._get_session_vector_data(session_id),
            performance_stats=performance_stats,
            export_timestamp=datetime.now().isoformat()
        )
        
        # Create export file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"{metadata.name}_{timestamp}"
        
        if export_format == "json":
            export_path = self.export_dir / f"{export_filename}.json"
            with open(export_path, 'w', encoding='utf-8') as f:
                # Export data'yı JSON serializable hale getir
                export_dict = self._export_data_to_dict(export_data)
                json.dump(export_dict, f, indent=2, ensure_ascii=False)
                
        elif export_format == "zip":
            export_path = self.export_dir / f"{export_filename}.zip"
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add session data as JSON
                export_dict = self._export_data_to_dict(export_data)
                zipf.writestr("session_data.json",
                            json.dumps(export_dict, indent=2, ensure_ascii=False))
                
                # Add vector files if they exist
                vector_base_path = Path(f"data/vector_db/sessions/{session_id}")
                vector_files = [
                    f"{vector_base_path}.index",
                    f"{vector_base_path}.chunks",
                    f"{vector_base_path}.meta.jsonl"
                ]
                
                for file_path in vector_files:
                    if Path(file_path).exists():
                        zipf.write(file_path, f"vector_data/{Path(file_path).name}")
        
        # Log activity
        self._log_activity(
            session_id, "export",
            f"Session exported as {export_format}",
            metadata={'export_path': str(export_path), 'format': export_format}
        )
        
        self.logger.info(f"Exported session {session_id} to {export_path}")
        return str(export_path)
    
    def get_session_analytics(self, session_id: Optional[str] = None,
                            created_by: Optional[str] = None, 
                            days: int = 30) -> Dict[str, Any]:
        """Oturum analitiklerini getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            if session_id:
                # Belirli bir oturum için
                cursor.execute("""
                    SELECT COUNT(*) as activity_count
                    FROM session_activity 
                    WHERE session_id = ? AND timestamp > ?
                """, (session_id, since_date))
                activity_stats = cursor.fetchone()
                
                metadata = self.get_session_metadata(session_id)
                return {
                    "session_metadata": asdict(metadata) if metadata else None,
                    "activity_count": activity_stats[0] if activity_stats else 0,
                    "performance_stats": self._get_session_performance_stats(session_id)
                }
            else:
                # Genel istatistikler
                base_query = "SELECT COUNT(*) as count FROM sessions WHERE created_at > ?"
                base_params = [since_date]
                
                if created_by:
                    base_query += " AND created_by = ?"
                    base_params.append(created_by)
                
                cursor.execute(base_query, base_params)
                total_sessions = cursor.fetchone()[0]
                
                # Status breakdown
                status_query = base_query.replace("COUNT(*) as count", "status, COUNT(*) as count") + " GROUP BY status"
                cursor.execute(status_query, base_params)
                status_breakdown = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Category breakdown  
                category_query = base_query.replace("COUNT(*) as count", "category, COUNT(*) as count") + " GROUP BY category"
                cursor.execute(category_query, base_params)
                category_breakdown = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_sessions": total_sessions,
                    "status_breakdown": status_breakdown,
                    "category_breakdown": category_breakdown,
                    "period_days": days
                }
    
    def _session_exists(self, name: str, created_by: str) -> bool:
        """Aynı isimde oturum var mı kontrol et"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM sessions WHERE name = ? AND created_by = ?", 
                (name, created_by)
            )
            return cursor.fetchone() is not None
    
    def _session_exists_by_id(self, session_id: str) -> bool:
        """ID ile oturum var mı kontrol et"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            return cursor.fetchone() is not None
    
    def _generate_session_id(self, name: str, created_by: str) -> str:
        """Benzersiz oturum ID'si oluştur"""
        base = f"{name}_{created_by}_{datetime.now().timestamp()}"
        return hashlib.md5(base.encode()).hexdigest()
    
    def _row_to_metadata(self, row) -> SessionMetadata:
        """Database row'unu SessionMetadata'ya çevir"""
        return SessionMetadata(
            session_id=row['session_id'],
            name=row['name'],
            description=row['description'] or "",
            category=SessionCategory(row['category']),
            status=SessionStatus(row['status']),
            created_by=row['created_by'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            last_accessed=row['last_accessed'],
            grade_level=row['grade_level'] or "",
            subject_area=row['subject_area'] or "",
            learning_objectives=json.loads(row['learning_objectives'] or '[]'),
            tags=json.loads(row['tags'] or '[]'),
            document_count=row['document_count'],
            total_chunks=row['total_chunks'],
            query_count=row['query_count'],
            avg_response_time=row['avg_response_time'],
            user_rating=row['user_rating'],
            notes=row['notes'] or "",
            is_public=bool(row['is_public']),
            collaborators=json.loads(row['collaborators'] or '[]'),
            backup_count=row['backup_count']
        )
    
    def _log_activity(self, session_id: str, activity_type: str, description: str,
                     user_id: Optional[str] = None, metadata: Dict = None):
        """Oturum etkinliğini logla"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_activity 
                (session_id, activity_type, description, timestamp, user_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id, activity_type, description, datetime.now().isoformat(),
                user_id, json.dumps(metadata or {})
            ))
    
    def _get_session_performance_stats(self, session_id: str) -> Dict[str, Any]:
        """Oturum performans istatistiklerini getir"""
        # Bu fonksiyon vector store ve experiment database ile entegre edilebilir
        return {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "last_activity": None,
            "document_count": 0,
            "chunk_count": 0
        }
    
    def _get_session_vector_data(self, session_id: str) -> Dict[str, Any]:
        """Oturum vektör verilerini getir"""
        vector_base_path = Path(f"data/vector_db/sessions/{session_id}")
        
        return {
            "has_index": (Path(f"{vector_base_path}.index")).exists(),
            "has_chunks": (Path(f"{vector_base_path}.chunks")).exists(), 
            "has_metadata": (Path(f"{vector_base_path}.meta.jsonl")).exists(),
            "vector_path": str(vector_base_path)
        }
    
    def _cleanup_old_backups(self, days: int = 30):
        """Eski yedekleri temizle"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT backup_path FROM session_backups 
                    WHERE created_at < ? AND auto_created = 1
                """, (cutoff_date.isoformat(),))
                
                old_backups = cursor.fetchall()
                
                for backup in old_backups:
                    backup_path = Path(backup[0])
                    if backup_path.exists():
                        backup_path.unlink()
                
                # Remove from database
                cursor.execute("""
                    DELETE FROM session_backups 
                    WHERE created_at < ? AND auto_created = 1
                """, (cutoff_date.isoformat(),))
                
                if old_backups:
                    self.logger.info(f"Cleaned up {len(old_backups)} old backups")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
    
    def _metadata_to_dict(self, metadata: SessionMetadata) -> Dict[str, Any]:
        """SessionMetadata'yı JSON serializable dict'e çevir"""
        metadata_dict = asdict(metadata)
        
        # Enum'ları string'e çevir
        if 'category' in metadata_dict:
            metadata_dict['category'] = metadata.category.value
        if 'status' in metadata_dict:
            metadata_dict['status'] = metadata.status.value
            
        return metadata_dict
    
    def _export_data_to_dict(self, export_data: SessionExportData) -> Dict[str, Any]:
        """SessionExportData'yı JSON serializable dict'e çevir"""
        export_dict = asdict(export_data)
        
        # Metadata içindeki enum'ları çevir
        if 'metadata' in export_dict:
            metadata_dict = export_dict['metadata']
            if isinstance(metadata_dict.get('category'), SessionCategory):
                metadata_dict['category'] = metadata_dict['category'].value
            if isinstance(metadata_dict.get('status'), SessionStatus):
                metadata_dict['status'] = metadata_dict['status'].value
        
        return export_dict


# Global instance
professional_session_manager = ProfessionalSessionManager()
#!/usr/bin/env python3
"""
Eski Ollama embedding'leriyle oluşturulmuş session'ları ve cache'leri temizler
"""

import os
import shutil
from pathlib import Path
import sys

def clean_cache():
    """Cache klasörünü temizle"""
    cache_dir = Path("data/cache")
    if cache_dir.exists():
        file_count = len(list(cache_dir.glob("*.cache")))
        print(f"🗑️ {file_count} cache dosyası temizleniyor...")
        shutil.rmtree(cache_dir)
        print("✅ Cache temizlendi")
    else:
        print("ℹ️ Cache klasörü bulunamadı")

def list_sessions():
    """Mevcut session'ları listele"""
    session_dir = Path("data/vector_db/sessions")
    if not session_dir.exists():
        print("ℹ️ Session klasörü bulunamadı")
        return []
    
    sessions = []
    for file_path in session_dir.glob("*"):
        if file_path.is_file() and not file_path.name.startswith('.'):
            # Session dosyalarını grupla (base name)
            base_name = file_path.stem.split('.')[0]
            if base_name not in sessions:
                sessions.append(base_name)
    
    return sessions

def backup_sessions():
    """Session'ları yedekle"""
    backup_dir = Path("data/backups/pre_cleanup")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    session_dir = Path("data/vector_db/sessions")
    if session_dir.exists():
        print("💾 Session'lar yedekleniyor...")
        shutil.copytree(session_dir, backup_dir / "sessions", dirs_exist_ok=True)
        print(f"✅ Session'lar yedeklendi: {backup_dir / 'sessions'}")

def clean_sessions(session_names=None):
    """Belirli session'ları temizle"""
    session_dir = Path("data/vector_db/sessions")
    if not session_dir.exists():
        return
    
    if session_names is None:
        # Tüm session'ları temizle
        print("🗑️ Tüm session'lar temizleniyor...")
        for file_path in session_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        print("✅ Tüm session'lar temizlendi")
    else:
        # Belirli session'ları temizle
        for session_name in session_names:
            files_removed = 0
            for ext in ['.index', '.chunks', '.meta.jsonl']:
                file_path = session_dir / f"{session_name}{ext}"
                if file_path.exists():
                    file_path.unlink()
                    files_removed += 1
            if files_removed > 0:
                print(f"✅ Session temizlendi: {session_name} ({files_removed} dosya)")

def main():
    print("🧹 RAG System Session & Cache Temizleyici")
    print("=" * 50)
    
    # Mevcut durumu göster
    sessions = list_sessions()
    cache_dir = Path("data/cache")
    cache_count = len(list(cache_dir.glob("*.cache"))) if cache_dir.exists() else 0
    
    print(f"📊 Mevcut Durum:")
    print(f"   • {len(sessions)} session: {', '.join(sessions)}")
    print(f"   • {cache_count:,} cache dosyası")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # Otomatik temizlik
        print("🤖 Otomatik temizlik başlıyor...")
        backup_sessions()
        clean_cache()
        clean_sessions()
        print("✨ Temizlik tamamlandı!")
        return
    
    # İnteraktif mod
    print("Seçenekler:")
    print("1. Sadece cache'i temizle (önerilen)")
    print("2. Cache + tüm session'ları temizle") 
    print("3. Belirli session'ları temizle")
    print("4. Sadece yedek al, temizleme")
    print("5. İptal")
    
    choice = input("\nSeçiminiz (1-5): ").strip()
    
    if choice == "1":
        clean_cache()
        print("✨ Cache temizliği tamamlandı!")
        
    elif choice == "2":
        backup_sessions()
        clean_cache() 
        clean_sessions()
        print("✨ Tam temizlik tamamlandı!")
        
    elif choice == "3":
        if not sessions:
            print("❌ Temizlenecek session bulunamadı")
            return
            
        print("\nMevcut session'lar:")
        for i, session in enumerate(sessions, 1):
            print(f"  {i}. {session}")
        
        selection = input(f"Temizlenecek session'ları seçin (1-{len(sessions)}, virgülle ayırın): ")
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_sessions = [sessions[i] for i in indices if 0 <= i < len(sessions)]
            
            if selected_sessions:
                backup_sessions()
                clean_cache()
                clean_sessions(selected_sessions)
                print("✨ Seçili session'lar temizlendi!")
            else:
                print("❌ Geçersiz seçim")
        except ValueError:
            print("❌ Geçersiz format")
            
    elif choice == "4":
        backup_sessions()
        print("✨ Yedekleme tamamlandı!")
        
    elif choice == "5":
        print("❌ İptal edildi")
        
    else:
        print("❌ Geçersiz seçim")

if __name__ == "__main__":
    main()
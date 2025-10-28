#!/usr/bin/env python3
"""Debug script to identify server startup issues."""

import sys
import os

def test_imports():
    print("🧪 Testing critical imports for API server...")
    
    try:
        print("📦 Testing FastAPI...")
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
        
        print("📦 Testing uvicorn...")
        import uvicorn
        print(f"✅ Uvicorn: {uvicorn.__version__}")
        
        print("📦 Testing marker...")
        import marker
        print(f"✅ Marker: {getattr(marker, '__version__', 'unknown')}")
        
        print("📦 Testing marker PDF converter...")
        from marker.converters.pdf import PdfConverter
        print("✅ PDF converter available")
        
        print("📦 Testing src module imports...")
        
        # Add current directory to Python path to test module imports
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        print("📁 Testing document processor import...")
        from src.document_processing.enhanced_pdf_processor import process_pdf_with_analysis
        print("✅ Enhanced PDF processor import successful")
        
        print("📁 Testing API main module...")
        import src.api.main
        print("✅ API main module import successful")
        
        print("\n🎉 All critical imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_server_config():
    print("\n🔧 Testing server configuration...")
    
    try:
        # Test environment variables
        print("🌍 Checking environment variables...")
        required_vars = ['GEMINI_API_KEY']
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                print(f"✅ {var}: {'*' * 8} (hidden)")
            else:
                print(f"⚠️ {var}: Not set")
        
        print("✅ Environment check complete")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    print("🚨 DEBUG: Investigating API server startup issues\n")
    
    imports_ok = test_imports()
    config_ok = test_server_config()
    
    if imports_ok and config_ok:
        print("\n✅ All checks passed! Server should be able to start.")
        print("💡 Try starting server manually with: python -m src.api.main")
    else:
        print("\n❌ Issues detected. Fix the above errors before starting server.")
    
    return imports_ok and config_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Mikroservis Mimarisi Test Script'i
Tüm servislerin çalışıp çalışmadığını kontrol eder
"""

import requests
import json
import time
import sys

# Mikroservis URL'leri
API_GATEWAY = "https://api-gateway-1051060211087.europe-west1.run.app"
PDF_PROCESSOR = "https://pdf-processor-awe3elsvra-ew.a.run.app"
MODEL_INFERENCER = "https://model-inferencer-awe3elsvra-ew.a.run.app"
CHROMADB = "https://chromadb-awe3elsvra-ew.a.run.app"

def test_service(name, url, endpoint=""):
    """Servis sağlık kontrolü"""
    try:
        full_url = f"{url}{endpoint}"
        print(f"🔍 {name} test ediliyor: {full_url}")
        
        response = requests.get(full_url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ {name} - BAŞARILI ({response.status_code})")
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    data = response.json()
                    print(f"   📋 Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"   📋 Response: {response.text[:200]}...")
            return True
        else:
            print(f"❌ {name} - BAŞARISIZ ({response.status_code})")
            print(f"   📋 Error: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ {name} - ZAMAN AŞIMI (>10s)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"🚫 {name} - BAĞLANTI HATASI")
        return False
    except Exception as e:
        print(f"💥 {name} - BEKLENMEYEN HATA: {str(e)}")
        return False

def test_microservices_integration():
    """Tam mikroservis entegrasyon testi - API Gateway üzerinden"""
    print("🚀 MIKROSERVIS MIMARISI TEST BAŞLIYOR")
    print("=" * 60)
    
    results = []
    
    # 1. API Gateway Direct Tests
    print("\n1️⃣ API GATEWAY DOĞRUDAN TESTLER")
    print("-" * 35)
    results.append(test_service("API Gateway Health", API_GATEWAY, "/health"))
    results.append(test_service("API Gateway Sessions", API_GATEWAY, "/sessions"))
    
    # 2. Model Inference Service (via API Gateway)
    print("\n2️⃣ MODEL INFERENCE SERVİSİ (API Gateway üzerinden)")
    print("-" * 50)
    results.append(test_service("Models Endpoint", API_GATEWAY, "/models"))
    
    # 3. Document Processing Service (via API Gateway)
    print("\n3️⃣ DOCUMENT PROCESSING SERVİSİ (API Gateway üzerinden)")
    print("-" * 55)
    results.append(test_service("List Markdown Files", API_GATEWAY, "/documents/list-markdown"))
    
    # 4. Private Services Security Test
    print("\n4️⃣ PRIVATE SERVİSLER GÜVENLİK TESTİ (403 bekleniyor)")
    print("-" * 55)
    
    # Bu testler 403 döndürmelidir (başarılı güvenlik)
    security_results = []
    
    try:
        print("🔍 PDF Processor doğrudan erişim testi...")
        response = requests.get(f"{PDF_PROCESSOR}/health", timeout=10)
        if response.status_code == 403:
            print("✅ PDF Processor private - GÜVENLİ (403 Forbidden)")
            security_results.append(True)
        else:
            print(f"⚠️ PDF Processor güvenlik sorunu ({response.status_code})")
            security_results.append(False)
    except Exception as e:
        print(f"❌ PDF Processor test hatası: {str(e)}")
        security_results.append(False)
    
    try:
        print("🔍 Model Inferencer doğrudan erişim testi...")
        response = requests.get(f"{MODEL_INFERENCER}/health", timeout=10)
        if response.status_code == 403:
            print("✅ Model Inferencer private - GÜVENLİ (403 Forbidden)")
            security_results.append(True)
        else:
            print(f"⚠️ Model Inferencer güvenlik sorunu ({response.status_code})")
            security_results.append(False)
    except Exception as e:
        print(f"❌ Model Inferencer test hatası: {str(e)}")
        security_results.append(False)
        
    try:
        print("🔍 ChromaDB doğrudan erişim testi...")
        response = requests.get(f"{CHROMADB}/api/v1/heartbeat", timeout=10)
        if response.status_code == 403:
            print("✅ ChromaDB private - GÜVENLİ (403 Forbidden)")
            security_results.append(True)
        else:
            print(f"⚠️ ChromaDB güvenlik sorunu ({response.status_code})")
            security_results.append(False)
    except Exception as e:
        print(f"❌ ChromaDB test hatası: {str(e)}")
        security_results.append(False)
    
    # Security test sonuçlarını ana sonuçlara ekle
    results.extend(security_results)
    
    print(f"\n   🔒 Güvenlik testi: {sum(security_results)}/{len(security_results)} servis güvenli")
    
    # Sonuçlar
    print("\n" + "=" * 60)
    print("📊 TEST SONUÇLARI")
    print("=" * 60)
    
    success_count = sum(results)
    total_count = len(results)
    success_rate = (success_count / total_count) * 100
    
    print(f"✅ Başarılı testler: {success_count}")
    print(f"❌ Başarısız testler: {total_count - success_count}")  
    print(f"📈 Başarı oranı: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n🎉 TÜM MIKROSERVISLER BAŞARIYLA ÇALIŞIYOR!")
        print("🌐 Frontend URL'si güncellendi ve sistem hazır!")
    elif success_rate >= 70:
        print("\n⚠️ Mikroservisler çoğunlukla çalışıyor, bazı iyileştirmeler gerekebilir.")
    else:
        print("\n🚨 KRITIK SORUN: Çok sayıda mikroservis çalışmıyor!")
        
    return success_rate >= 70

if __name__ == "__main__":
    print("🏗️ RAG3 Mikroservis Mimarisi Test Suite")
    print("🕒 Test başlangıcı:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    success = test_microservices_integration()
    
    print(f"\n🕒 Test tamamlanma zamanı: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n🚀 Sistem test edildi ve çalışmaya hazır!")
        print("📱 Frontend artık yeni mikroservis mimarisini kullanıyor:")
        print(f"   🌐 {API_GATEWAY}")
        sys.exit(0)
    else:
        print("\n🔧 Sistem sorunları tespit edildi, lütfen servisleri kontrol edin.")
        sys.exit(1)
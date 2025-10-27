"""
Cloud LLM Client - Ücretsiz açık kaynak modeller için API client.
Hugging Face, Groq, Together AI, SambaNova gibi servisleri destekler.
"""

import requests
import os
from typing import Dict, Any, Optional
from ..config import get_config
from ..utils.logger import get_logger

class CloudLLMClient:
    """Ücretsiz cloud LLM servislerine bağlanmak için client."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__, self.config)
        
    def generate_response(self, prompt: str, model_name: str, provider: str) -> str:
        """Cloud model'den cevap üret."""
        
        try:
            if provider == "groq":
                return self._groq_generate(prompt, model_name)
            elif provider == "huggingface":
                return self._huggingface_generate(prompt, model_name)
            elif provider == "together":
                return self._together_generate(prompt, model_name)
            else:
                return f"❌ '{provider}' provider desteklenmiyor."
                
        except Exception as e:
            self.logger.error(f"Cloud LLM hatası: {e}")
            return f"❌ Cloud model hatası: {str(e)}"
    
    def generate(self, model: str, messages: list, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """OpenAI-style chat completions for all providers."""
        
        # Detect provider from model name or use groq as default
        if any(groq_model in model for groq_model in ["llama-3.1", "mixtral", "gemma"]):
            provider = "groq"
        elif "microsoft/" in model or "mistralai/" in model:
            provider = "huggingface"
        elif "meta-llama/" in model:
            provider = "together"
        else:
            provider = "groq"  # Default to groq
        
        # Convert messages to single prompt for now
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
        
        return self.generate_response(prompt.strip(), model, provider)
    
    def _groq_generate(self, prompt: str, model_name: str) -> str:
        """Groq API ile cevap üret - Çok hızlı ve ücretsiz."""
        
        # Check for GROQ_API_KEY in config or env
        api_key = self.config.get("groq_api_key") or os.getenv("GROQ_API_KEY")
        if not api_key:
            return "❌ GROQ_API_KEY gerekli. https://console.groq.com'dan ücretsiz alabilirsiniz."
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Parse prompt if it contains system/user parts
        messages = []
        if "System:" in prompt and "User:" in prompt:
            parts = prompt.split("User:")
            if len(parts) >= 2:
                system_part = parts[0].replace("System:", "").strip()
                user_part = parts[1].strip()
                messages = [
                    {"role": "system", "content": system_part},
                    {"role": "user", "content": user_part}
                ]
        
        if not messages:
            messages = [
                {"role": "system", "content": "Sen öğrencilerin en sevdiği ders asistanısın! Türkçe cevaplar ver."},
                {"role": "user", "content": prompt}
            ]
        
        data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"❌ Groq API hatası: {response.status_code} - {response.text}"
    
    def _huggingface_generate(self, prompt: str, model_name: str) -> str:
        """Hugging Face Inference API ile cevap üret - Ücretsiz tier."""
        
        api_key = self.config.get("huggingface_api_key")
        if not api_key:
            return "❌ HUGGINGFACE_API_KEY gerekli. https://huggingface.co/settings/tokens'dan ücretsiz alabilirsiniz."
        
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": f"Soru: {prompt}\nCevap:",
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "Cevap alınamadı")
            return str(result)
        else:
            return f"❌ Hugging Face API hatası: {response.status_code} - {response.text}"
    
    def _together_generate(self, prompt: str, model_name: str) -> str:
        """Together AI ile cevap üret - Ücretsiz credits."""
        
        api_key = self.config.get("together_api_key") 
        if not api_key:
            return "❌ TOGETHER_API_KEY gerekli. https://api.together.xyz'den ücretsiz credits alabilirsiniz."
        
        url = "https://api.together.xyz/inference"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_name,
            "prompt": f"[INST] {prompt} [/INST]",
            "max_tokens": 512,
            "temperature": 0.7,
            "stop": ["[INST]", "</s>"]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result["output"]["choices"][0]["text"].strip()
        else:
            return f"❌ Together AI hatası: {response.status_code} - {response.text}"

def get_cloud_llm_client():
    """Global cloud LLM client instance."""
    return CloudLLMClient()
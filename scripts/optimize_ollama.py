#!/usr/bin/env python3
"""
Ollama GPU optimization script.
This script configures Ollama for optimal GPU usage.
"""

import os
import subprocess
import sys
import time

def run_command(command, timeout=60):
    """Run a command with timeout."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {command}")
        return False, "", "Timeout"

def set_ollama_env_vars():
    """Set Ollama environment variables for GPU optimization."""
    env_vars = {
        'OLLAMA_NUM_GPU': '1',
        'OLLAMA_GPU_LAYERS': '-1',  # Use all available GPU layers
        'OLLAMA_LOAD_TIMEOUT': '300',
        'OLLAMA_REQUEST_TIMEOUT': '120',
        'OLLAMA_HOST': '127.0.0.1:11434',
        'OLLAMA_MODELS': os.path.expanduser('~/.ollama/models'),
        'CUDA_VISIBLE_DEVICES': '0'
    }
    
    print("Setting Ollama environment variables for GPU optimization...")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")

def check_ollama_service():
    """Check if Ollama service is running."""
    print("Checking Ollama service status...")
    success, stdout, stderr = run_command("ollama list", timeout=30)
    
    if success:
        print("✓ Ollama service is running")
        print(f"Available models:\n{stdout}")
        return True
    else:
        print("✗ Ollama service is not running or not responding")
        print(f"Error: {stderr}")
        return False

def restart_ollama_service():
    """Restart Ollama service to apply new environment variables."""
    print("Restarting Ollama service to apply GPU optimizations...")
    
    # Try to stop existing Ollama processes
    print("Stopping existing Ollama processes...")
    run_command("taskkill /f /im ollama.exe", timeout=10)
    time.sleep(3)
    
    # Start Ollama with optimized settings
    print("Starting Ollama with GPU optimizations...")
    
    # Set environment and start Ollama
    env = os.environ.copy()
    env.update({
        'OLLAMA_NUM_GPU': '1',
        'OLLAMA_GPU_LAYERS': '-1',
        'OLLAMA_LOAD_TIMEOUT': '300',
        'OLLAMA_REQUEST_TIMEOUT': '120',
        'CUDA_VISIBLE_DEVICES': '0'
    })
    
    try:
        # Start Ollama serve in the background
        subprocess.Popen(['ollama', 'serve'], env=env)
        print("Ollama service started with GPU optimizations")
        
        # Wait a bit for service to start
        time.sleep(5)
        
        return check_ollama_service()
        
    except Exception as e:
        print(f"Failed to start Ollama service: {e}")
        return False

def test_model_performance():
    """Test model performance with a simple query."""
    print("Testing model performance...")
    
    test_query = "Merhaba, bu bir performans testidir. GPU kullanıyor musun?"
    command = f'ollama run qwen2.5:14b "{test_query}"'
    
    start_time = time.time()
    success, stdout, stderr = run_command(command, timeout=60)
    end_time = time.time()
    
    if success:
        response_time = end_time - start_time
        print(f"✓ Model response received in {response_time:.2f} seconds")
        print(f"Response: {stdout[:200]}...")
        return True
    else:
        print(f"✗ Model test failed: {stderr}")
        return False

def main():
    """Main optimization function."""
    print("=== Ollama GPU Optimization Script ===\n")
    
    # Step 1: Set environment variables
    set_ollama_env_vars()
    print()
    
    # Step 2: Check current service status
    if not check_ollama_service():
        print("\nAttempting to restart Ollama service...")
        if not restart_ollama_service():
            print("Failed to start Ollama service. Please start it manually.")
            return False
    print()
    
    # Step 3: Test performance
    success = test_model_performance()
    print()
    
    if success:
        print("✓ Optimization completed successfully!")
        print("\nRecommendations:")
        print("1. Monitor GPU usage with: nvidia-smi")
        print("2. Check Ollama logs if you encounter issues")
        print("3. Restart Ollama service if performance degrades")
    else:
        print("✗ Optimization completed with issues")
        print("Please check Ollama installation and GPU drivers")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
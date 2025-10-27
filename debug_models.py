from src.utils.model_selector import get_available_models_info

print("=== AVAILABLE MODELS DEBUG ===")
models = get_available_models_info()

print(f"Total models: {len(models)}")
print()

for k, v in models.items():
    print(f"Model: {k}")
    print(f"  Type: {v.get('type')}")
    print(f"  Provider: {v.get('provider')}")
    print(f"  Installed: {v.get('installed')}")
    print(f"  Status: {v.get('status')}")
    if 'cloud' in k.lower():
        print("  ^^^^ CLOUD MODEL DETECTED ^^^^")
    print()

print("=== CLOUD MODELS ONLY ===")
cloud_models = {k: v for k, v in models.items() if 'cloud' in k.lower() or v.get('provider') == 'cloud'}
print(f"Cloud models found: {len(cloud_models)}")
for k, v in cloud_models.items():
    print(f"  {k}: provider={v.get('provider')}, installed={v.get('installed')}")
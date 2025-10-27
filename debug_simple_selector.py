from src.utils.model_selector import get_available_models_info

print("=== SIMPLE MODEL SELECTOR DEBUG ===")

# Same logic as create_simple_model_selector_ui()
available_models = get_available_models_info()

# TÜM modelleri göster (hem yerel hem cloud)
usable_models = {}

for k, v in available_models.items():
    # Yerel modeller: yüklü olanlar
    # Cloud modeller: her zaman kullanılabilir
    condition1 = v.get('installed', False)
    condition2 = v.get('provider') in ['groq', 'huggingface', 'together', 'cloud']
    
    print(f"Model: {k}")
    print(f"  installed: {condition1}")
    print(f"  provider check: {condition2} (provider={v.get('provider')})")
    print(f"  RESULT: {condition1 or condition2}")
    
    if condition1 or condition2:
        usable_models[k] = v
        print(f"  ^^^ ADDED TO USABLE MODELS ^^^")
    else:
        print(f"  XXX FILTERED OUT XXX")
    print()

print("=== FINAL USABLE MODELS ===")
print(f"Total usable models: {len(usable_models)}")
for k, v in usable_models.items():
    provider_icon = ""
    if v.get('provider') == 'groq':
        provider_icon = "🌐"
    elif v.get('provider') == 'huggingface':
        provider_icon = "🤗"
    elif v.get('provider') == 'together':
        provider_icon = "🤝"
    elif 'cloud' in k.lower():
        provider_icon = "☁️"
    else:
        provider_icon = "🏠"
    
    label = f"{provider_icon} {v['name']}"
    print(f"  {label}")
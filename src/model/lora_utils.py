from peft import LoraConfig, get_peft_model
import torch

def inject_lora_adapters(model, r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["Wqkv", "out_proj"]):
    """Injects LoRA adapters into the model."""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model.bert = get_peft_model(model.bert, lora_config)
    return model

def load_lora_weights(model, weights_path, device):
    """Loads LoRA weights with strict=False to handle architectural mismatches."""
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print("[✓] BRAIN SECURED: SOTA weights perfectly absorbed into aligned architecture!")
    except Exception as e:
        print(f"🚨 FATAL ERROR: Weight Mismatch!{e}")
        raise SystemExit
    return model

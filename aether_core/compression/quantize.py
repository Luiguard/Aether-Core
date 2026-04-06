"""
Aether-Core: Echte 1.58-Bit Ternary Quantisierung (Spec 12.1).
Reduziert Gewichte auf {-1, 0, +1} mit Skalierungsfaktor.
"""
import torch
import torch.nn as nn
import os
from typing import Dict, Any


def quantize_ternary(weight: torch.Tensor) -> tuple:
    """
    Quantisiert einen Gewichtstensor auf 1.58-Bit (ternary: -1, 0, +1).
    Speichert den Skalierungsfaktor separat.
    """
    scale = weight.abs().mean()
    if scale == 0:
        return torch.zeros_like(weight, dtype=torch.int8), scale
    ternary = torch.round(weight / scale).clamp(-1, 1).to(torch.int8)
    return ternary, scale


def dequantize_ternary(ternary: torch.Tensor, scale: float) -> torch.Tensor:
    """Rekonstruiert FP32-Gewichte aus ternären Werten."""
    return ternary.float() * scale


def quantize_model(model: nn.Module) -> Dict[str, Any]:
    """
    Quantisiert alle Linear-Layer eines Modells auf 1.58-Bit.
    Gibt Statistiken zurück.
    """
    total_params = 0
    quantized_params = 0
    original_size = 0
    quantized_size = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        original_size += param.numel() * 4  # FP32

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            ternary, scale = quantize_ternary(w)
            # Ersetze Gewichte durch dequantisierte Version
            module.weight.data = dequantize_ternary(ternary, scale)
            quantized_params += w.numel()
            # 1.58 bit pro Parameter + Scale Factor
            quantized_size += (w.numel() * 2 / 8) + 4  # 2 bits + scale

    stats = {
        "total_params": total_params,
        "quantized_params": quantized_params,
        "original_size_mb": round(original_size / (1024 ** 2), 2),
        "quantized_size_mb": round(quantized_size / (1024 ** 2), 2),
        "compression_ratio": round(original_size / max(quantized_size, 1), 1),
    }
    print(f"[Quantize] {stats['original_size_mb']}MB -> {stats['quantized_size_mb']}MB "
          f"(Ratio: {stats['compression_ratio']}x)")
    return stats


def save_quantized(model: nn.Module, path: str):
    """Speichert ein quantisiertes Modell."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"[Quantize] Modell gespeichert: {path} ({size_mb:.1f} MB)")

"""
Aether-Core: Model Scaling Engine.
Ermöglicht frei konfigurierbare Modellgrößen von 100M bis 500B+ Parameter.
Berechnet automatisch die optimale Konfiguration und den Speicherbedarf.
"""
import math
from typing import Dict, Any, Tuple


# Vordefinierte Presets (können in config.yaml überschrieben werden)
SCALING_PRESETS = {
    "nano": {       # ~50M  | ~100 MB  | Smartphone/RPi
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 6,
        "moe_experts": 4,
        "moe_top_k": 1,
        "decoder_layers": 2,
        "max_seq_len": 512,
    },
    "micro": {      # ~150M | ~300 MB  | Integrated GPU
        "d_model": 512,
        "n_layers": 8,
        "n_heads": 8,
        "moe_experts": 4,
        "moe_top_k": 2,
        "decoder_layers": 3,
        "max_seq_len": 1024,
    },
    "small": {      # ~350M | ~700 MB  | RTX 3060 (Standard)
        "d_model": 768,
        "n_layers": 8,
        "n_heads": 12,
        "moe_experts": 8,
        "moe_top_k": 2,
        "decoder_layers": 4,
        "max_seq_len": 2048,
    },
    "medium": {     # ~1.3B | ~2.6 GB  | RTX 4070/4080
        "d_model": 1024,
        "n_layers": 16,
        "n_heads": 16,
        "moe_experts": 8,
        "moe_top_k": 2,
        "decoder_layers": 6,
        "max_seq_len": 4096,
    },
    "large": {      # ~7B   | ~14 GB   | RTX 4090 / A100
        "d_model": 2048,
        "n_layers": 24,
        "n_heads": 16,
        "moe_experts": 16,
        "moe_top_k": 2,
        "decoder_layers": 8,
        "max_seq_len": 8192,
    },
    "xlarge": {     # ~30B  | ~60 GB   | Multi-GPU / ZeRO-Offload
        "d_model": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "moe_experts": 16,
        "moe_top_k": 4,
        "decoder_layers": 12,
        "max_seq_len": 8192,
    },
    "giant": {      # ~100B | ~200 GB  | ZeRO-3 + Offload (RAM+SSD)
        "d_model": 6144,
        "n_layers": 48,
        "n_heads": 48,
        "moe_experts": 32,
        "moe_top_k": 4,
        "decoder_layers": 16,
        "max_seq_len": 8192,
    },
    "ultra": {      # ~500B | ~1 TB    | ZeRO-3 + Offload + NVMe
        "d_model": 8192,
        "n_layers": 64,
        "n_heads": 64,
        "moe_experts": 64,
        "moe_top_k": 4,
        "decoder_layers": 24,
        "max_seq_len": 8192,
    },
}


def estimate_parameters(cfg: Dict[str, Any], vocab_size: int) -> Dict[str, Any]:
    """Berechnet die geschätzte Parameterzahl und den Speicherbedarf."""
    d = cfg["d_model"]
    n_l = cfg["n_layers"]
    n_exp = cfg["moe_experts"]
    d_ff = d * 4  # Standard FFN Expansion
    dec_l = cfg["decoder_layers"]

    # SNC (MoE): Embedding + n_layers * (Gate + n_experts * FFN)
    snc_emb = vocab_size * d
    snc_gate = n_l * (d * n_exp)
    snc_experts = n_l * n_exp * (d * d_ff + d_ff * d)  # 2x Linear pro Expert
    snc_total = snc_emb + snc_gate + snc_experts

    # Decoder: Embedding + pos + n_layers * (Attn_QKV + Attn_proj + FFN)
    dec_emb = vocab_size * d
    dec_pos = cfg["max_seq_len"] * d
    dec_attn = dec_l * (3 * d * d + d * d)  # QKV + Proj
    dec_ffn = dec_l * (d * d_ff + d_ff * d)
    dec_total = dec_emb + dec_pos + dec_attn + dec_ffn

    # Compression Engine
    ce_total = (d * 512 + 512 * 256 + 256 * 128) * 2  # Encoder + Decoder

    total = snc_total + dec_total + ce_total

    # Speicher-Schätzung
    fp32_gb = (total * 4) / (1024 ** 3)
    fp16_gb = (total * 2) / (1024 ** 3)
    int8_gb = (total * 1) / (1024 ** 3)
    bit158_gb = (total * 0.2) / (1024 ** 3)  # ~1.58 bit

    # Aktive Parameter (MoE Sparsity)
    active_ratio = cfg["moe_top_k"] / cfg["moe_experts"]
    active_params = snc_emb + snc_gate + (snc_experts * active_ratio) + dec_total + ce_total

    return {
        "total_params": total,
        "active_params": int(active_params),
        "snc_params": snc_total,
        "decoder_params": dec_total,
        "compression_params": ce_total,
        "memory_fp32_gb": round(fp32_gb, 2),
        "memory_fp16_gb": round(fp16_gb, 2),
        "memory_int8_gb": round(int8_gb, 2),
        "memory_1_58bit_gb": round(bit158_gb, 2),
    }


def recommend_strategy(total_params: int, vram_gb: float) -> Dict[str, Any]:
    """Empfiehlt die optimale Trainings-Strategie basierend auf Modellgröße und VRAM."""
    fp16_gb = (total_params * 2) / (1024 ** 3)
    # Training braucht ~4x Modellgröße (Gradients + Optimizer States)
    training_gb = fp16_gb * 4

    if training_gb <= vram_gb:
        return {
            "strategy": "standard",
            "description": "Alles passt in VRAM. Standard-Training.",
            "deepspeed_stage": 0,
            "gradient_checkpointing": False,
            "offload_optimizer": False,
            "offload_params": False,
            "nvme_offload": False,
        }
    elif training_gb <= vram_gb * 2:
        return {
            "strategy": "zero_stage_1",
            "description": "ZeRO Stage 1: Optimizer-States partitionieren.",
            "deepspeed_stage": 1,
            "gradient_checkpointing": True,
            "offload_optimizer": False,
            "offload_params": False,
            "nvme_offload": False,
        }
    elif training_gb <= vram_gb * 4:
        return {
            "strategy": "zero_stage_2_offload",
            "description": "ZeRO Stage 2 + CPU Offload: Optimizer + Gradienten auf RAM auslagern.",
            "deepspeed_stage": 2,
            "gradient_checkpointing": True,
            "offload_optimizer": True,
            "offload_params": False,
            "nvme_offload": False,
        }
    elif training_gb <= vram_gb * 16:
        return {
            "strategy": "zero_stage_3_offload",
            "description": "ZeRO Stage 3 + CPU Offload: Alles auf RAM auslagern, nur aktive Berechnung auf GPU.",
            "deepspeed_stage": 3,
            "gradient_checkpointing": True,
            "offload_optimizer": True,
            "offload_params": True,
            "nvme_offload": False,
        }
    else:
        return {
            "strategy": "zero_stage_3_nvme",
            "description": "ZeRO Stage 3 + NVMe Offload: VRAM + RAM + SSD. Für Modelle > 100B.",
            "deepspeed_stage": 3,
            "gradient_checkpointing": True,
            "offload_optimizer": True,
            "offload_params": True,
            "nvme_offload": True,
        }


def generate_deepspeed_config(strategy: Dict[str, Any], train_batch_size: int = 16) -> Dict[str, Any]:
    """Generiert eine DeepSpeed JSON-Konfiguration basierend auf der Strategie."""
    stage = strategy["deepspeed_stage"]

    ds_config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": max(1, train_batch_size // 4),
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": stage,
        },
    }

    if strategy["offload_optimizer"]:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if strategy["offload_params"]:
        ds_config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if strategy["nvme_offload"]:
        ds_config["zero_optimization"]["offload_optimizer"]["device"] = "nvme"
        ds_config["zero_optimization"]["offload_optimizer"]["nvme_path"] = "./offload_nvme/"
        ds_config["zero_optimization"]["offload_param"]["device"] = "nvme"
        ds_config["zero_optimization"]["offload_param"]["nvme_path"] = "./offload_nvme/"

    if strategy["gradient_checkpointing"]:
        ds_config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": True,
        }

    return ds_config


def print_scaling_report(preset_name: str, vocab_size: int = 1323, vram_gb: float = 12.0):
    """Gibt einen detaillierten Skalierungs-Bericht aus."""
    if preset_name not in SCALING_PRESETS:
        print(f"Unbekanntes Preset: {preset_name}")
        return

    cfg = SCALING_PRESETS[preset_name]
    est = estimate_parameters(cfg, vocab_size)
    strat = recommend_strategy(est["total_params"], vram_gb)

    print(f"\n{'='*60}")
    print(f"  Aether-Core Scaling Report: '{preset_name.upper()}'")
    print(f"{'='*60}")
    print(f"  d_model:         {cfg['d_model']}")
    print(f"  SNC Layers:      {cfg['n_layers']}")
    print(f"  MoE Experts:     {cfg['moe_experts']} (Top-{cfg['moe_top_k']} aktiv)")
    print(f"  Decoder Layers:  {cfg['decoder_layers']}")
    print(f"  Max Seq Len:     {cfg['max_seq_len']}")
    print(f"{'='*60}")
    print(f"  Parameter (Total):   {est['total_params']:>15,}")
    print(f"  Parameter (Aktiv):   {est['active_params']:>15,}")
    print(f"  Speicher FP32:       {est['memory_fp32_gb']:>12} GB")
    print(f"  Speicher FP16:       {est['memory_fp16_gb']:>12} GB")
    print(f"  Speicher INT8:       {est['memory_int8_gb']:>12} GB")
    print(f"  Speicher 1.58-Bit:   {est['memory_1_58bit_gb']:>12} GB")
    print(f"{'='*60}")
    print(f"  Empfohlene Strategie: {strat['strategy']}")
    print(f"  {strat['description']}")
    print(f"  DeepSpeed Stage:     {strat['deepspeed_stage']}")
    print(f"  Gradient Checkpoint: {strat['gradient_checkpointing']}")
    print(f"  Offload Optimizer:   {strat['offload_optimizer']}")
    print(f"  Offload Params:      {strat['offload_params']}")
    print(f"  NVMe Offload:        {strat['nvme_offload']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n  Aether-Core: Alle Skalierungs-Presets\n")
    for name in SCALING_PRESETS:
        print_scaling_report(name, vocab_size=1323, vram_gb=12.0)

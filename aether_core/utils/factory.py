"""
Aether-Core: Model Factory.
Baut alle Modell-Komponenten basierend auf dem gewählten Scaling-Preset.
Generiert automatisch die DeepSpeed-Konfiguration.
"""
import json
import torch
from typing import Dict, Any, Tuple

from aether_core.neural.moe import SparseCore
from aether_core.neural.decoder import ChatDecoder
from aether_core.compression.engine import CompressionEngine
from aether_core.utils.scaling import SCALING_PRESETS, estimate_parameters, recommend_strategy, generate_deepspeed_config


def build_models(config: Dict[str, Any], vocab_size: int, device: torch.device) -> Tuple:
    """
    Baut SNC, Decoder und CompressionEngine basierend auf der Config.
    Wendet das Scaling-Preset an, falls gesetzt.
    """
    # 1. Preset anwenden (falls vorhanden)
    preset_name = config.get("scaling", {}).get("preset", None)
    n_cfg = config["neural"]

    if preset_name and preset_name in SCALING_PRESETS:
        preset = SCALING_PRESETS[preset_name]
        n_cfg["d_model"] = preset["d_model"]
        n_cfg["n_layers"] = preset["n_layers"]
        n_cfg["n_heads"] = preset["n_heads"]
        n_cfg["moe"]["n_experts"] = preset["moe_experts"]
        n_cfg["moe"]["top_k"] = preset["moe_top_k"]
        n_cfg["max_seq_len"] = preset["max_seq_len"]
        decoder_layers = preset["decoder_layers"]
        print(f"[Factory] Scaling-Preset '{preset_name}' angewendet.")
    else:
        decoder_layers = 4
        print(f"[Factory] Kein Preset. Nutze manuelle Config.")

    c_cfg = config["compression"]

    # 2. Parameter-Schätzung
    est = estimate_parameters(
        SCALING_PRESETS.get(preset_name, {
            "d_model": n_cfg["d_model"],
            "n_layers": n_cfg["n_layers"],
            "n_heads": n_cfg["n_heads"],
            "moe_experts": n_cfg["moe"]["n_experts"],
            "moe_top_k": n_cfg["moe"]["top_k"],
            "decoder_layers": decoder_layers,
            "max_seq_len": n_cfg.get("max_seq_len", 2048),
        }),
        vocab_size,
    )
    print(f"[Factory] Parameter: {est['total_params']:,} (Aktiv: {est['active_params']:,})")
    print(f"[Factory] Speicher: FP16={est['memory_fp16_gb']}GB | INT8={est['memory_int8_gb']}GB | 1.58bit={est['memory_1_58bit_gb']}GB")

    # 3. Strategie empfehlen
    vram_gb = config.get("vram_gb", 12.0)
    strat = recommend_strategy(est["total_params"], vram_gb)
    print(f"[Factory] Strategie: {strat['strategy']} ({strat['description']})")

    # 4. DeepSpeed Config generieren (falls nötig)
    if config.get("deepspeed", {}).get("enabled", False) and strat["deepspeed_stage"] > 0:
        ds_cfg = generate_deepspeed_config(strat, config.get("training", {}).get("batch_size", 16))
        ds_path = config.get("deepspeed", {}).get("config_path", "ds_config.json")
        with open(ds_path, "w") as f:
            json.dump(ds_cfg, f, indent=2)
        print(f"[Factory] DeepSpeed Config generiert: {ds_path}")

    # 5. Modelle bauen
    snc = SparseCore(
        vocab_size=vocab_size,
        d_model=n_cfg["d_model"],
        n_layers=n_cfg["n_layers"],
        n_experts=n_cfg["moe"]["n_experts"],
        top_k=n_cfg["moe"]["top_k"],
    )

    decoder = ChatDecoder(
        vocab_size=vocab_size,
        d_model=n_cfg["d_model"],
        n_layers=decoder_layers,
        n_heads=n_cfg["n_heads"],
        max_seq_len=n_cfg.get("max_seq_len", 2048),
    )

    ce = CompressionEngine(
        input_dim=n_cfg["d_model"],
        latent_dim=c_cfg["latent_dim"],
    )

    # Gradient Checkpointing aktivieren (wenn empfohlen)
    if strat.get("gradient_checkpointing", False):
        if hasattr(decoder, "blocks"):
            for block in decoder.blocks:
                block.use_checkpoint = True
        print(f"[Factory] Gradient Checkpointing aktiviert.")

    # Auf Device verschieben
    snc = snc.to(device)
    decoder = decoder.to(device)
    ce = ce.to(device)

    return snc, decoder, ce, est, strat

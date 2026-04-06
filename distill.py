"""
Aether-Core: Knowledge Distillation Pipeline.
Trainiert den Student (SNC + Decoder) auf Basis von Teacher-generierten Daten.

Ablauf:
1. Teacher (DeepSeek) generiert Frage-Antwort-Paare.
2. Tokenizer kodiert diese Paare.
3. Student lernt, die Antworten des Teachers zu reproduzieren (CrossEntropy Loss).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys
import yaml
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether_core.neural.moe import SparseCore
from aether_core.neural.decoder import ChatDecoder
from aether_core.compression.engine import CompressionEngine
from aether_core.symbolic.symbolic_memory import SymbolicMemory
from aether_core.utils.tokenizer import AetherTokenizer
from aether_core.utils.teacher import TeacherClient
from aether_core.utils.checkpoint import CheckpointManager


def load_or_generate_data(teacher: TeacherClient, data_path: str, topics: list) -> list:
    """Lädt vorhandene Trainingsdaten oder generiert neue via Teacher."""
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"[Distill] Vorhandene Daten geladen: {len(pairs)} Paare.")
        return pairs

    if not teacher.api_key:
        print("[Distill] Kein API-Key. Generiere synthetische Offline-Daten...")
        # Fallback: Offline-Daten aus vorhandener Domäne
        pairs = []
        for t in topics:
            pairs.append({"question": t, "answer": f"Dies ist eine Erklärung zum Thema {t}."})
        return pairs

    pairs = teacher.generate_training_batch(topics)
    teacher.save_training_data(pairs, data_path)
    return pairs


def prepare_sequences(pairs: list, tokenizer: AetherTokenizer, max_len: int = 128):
    """Wandelt QA-Paare in Token-Sequenzen für das Training um."""
    sequences = []
    for pair in pairs:
        q = pair["question"]
        a = pair["answer"]
        text = f"User: {q}\nAssistant: {a}"
        ids = tokenizer.encode(text)

        # Auf max_len begrenzen oder auffüllen
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))

        sequences.append(ids)

    return torch.tensor(sequences, dtype=torch.long)


def distill(config_path: str = "config.yaml", custom_epochs: int = None):
    """Hauptfunktion: Distillation durchführen."""

    # 1. Config laden
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    n_cfg = config["neural"]
    c_cfg = config["compression"]
    t_cfg = config.get("training", {})

    # 2. Tokenizer (laden oder trainieren)
    merges_path = "aether_core/data/tokenizer_merges.json"
    tokenizer = AetherTokenizer(merges_path if os.path.exists(merges_path) else None)

    if tokenizer.vocab_size < 300:
        print("[Distill] Tokenizer hat zu wenig Vokabeln. Trainiere auf Fallback-Korpus...")
        from aether_core.utils.train_tokenizer import FALLBACK_CORPUS
        tokenizer.train_bpe(FALLBACK_CORPUS, num_merges=8000)
        tokenizer.save_merges(merges_path)

    print(f"[Distill] Tokenizer: {tokenizer.vocab_size} Tokens")

    # 3. Modelle initialisieren
    decoder = ChatDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=n_cfg["d_model"],
        n_layers=4,
        n_heads=n_cfg.get("n_heads", 12),
    ).to(device)

    snc = SparseCore(
        vocab_size=tokenizer.vocab_size,
        d_model=n_cfg["d_model"],
        n_layers=n_cfg["n_layers"],
        n_experts=n_cfg["moe"]["n_experts"],
        top_k=n_cfg["moe"]["top_k"],
    ).to(device)

    ce = CompressionEngine(
        input_dim=n_cfg["d_model"],
        latent_dim=c_cfg["latent_dim"],
    ).to(device)

    # 4. Checkpoint laden falls vorhanden
    ckpt_mgr = CheckpointManager(t_cfg.get("output_dir", "checkpoints"))
    latest = ckpt_mgr.find_latest()
    start_step = 0
    if latest:
        start_step = ckpt_mgr.load(latest, snc, decoder, ce)
        print(f"[Checkpoint] Fortsetzung ab Step {start_step}")

    # 5. Teacher + Trainingsdaten
    teacher = TeacherClient()
    topics = [
        "Was ist ein Mixture-of-Experts Modell?",
        "Wie funktioniert Sparsity in neuronalen Netzen?",
        "Was ist ein Transformer?",
        "Erkläre Self-Attention.",
        "Was ist Knowledge Distillation?",
        "Was ist Quantisierung?",
        "Was ist ein Wissensgraph?",
        "Was ist ein Autoencoder?",
        "Was ist Beam Search?",
        "Was ist ein Tokenizer?",
        "Was ist VRAM und warum ist es ein Engpass?",
        "Wie funktioniert ein Decoder in einem LLM?",
        "Was ist L1-Regularisierung?",
        "Was ist Mixed-Precision Training?",
        "Was ist der Unterschied zwischen Inference und Training?",
    ]

    data_path = "aether_core/data/training_data.json"
    pairs = load_or_generate_data(teacher, data_path, topics)

    # 6. Daten vorbereiten
    max_len = min(128, n_cfg.get("max_seq_len", 2048))
    data_tensor = prepare_sequences(pairs, tokenizer, max_len).to(device)
    print(f"[Distill] Trainings-Tensor: {data_tensor.shape}")

    # 7. Training
    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(snc.parameters()),
        lr=t_cfg.get("learning_rate", 1e-4),
    )

    batch_size = min(t_cfg.get("batch_size", 4), len(data_tensor))
    epochs = custom_epochs if custom_epochs is not None else t_cfg.get("epochs", 20)
    save_every = t_cfg.get("save_every", 500)

    decoder.train()
    snc.train()

    global_step = start_step
    print(f"\n[Distill] === Starte Training (Epochs: {epochs}, Batch: {batch_size}) ===\n")

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(data_tensor.size(0))
        data_tensor = data_tensor[perm]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i : i + batch_size]

            # Input: alle Tokens außer dem letzten
            input_ids = batch[:, :-1]
            # Target: alle Tokens außer dem ersten (shifted)
            target_ids = batch[:, 1:]

            # Forward: Decoder (autoregressive Logits)
            logits = decoder(input_ids)  # [batch, seq_len-1, vocab]

            # Loss: CrossEntropy (ignoriere Padding)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % save_every == 0:
                ckpt_mgr.save(global_step, snc, decoder, ce, optimizer)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[Distill] Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Step: {global_step}")

    # 8. Finaler Checkpoint
    ckpt_mgr.save(global_step, snc, decoder, ce, optimizer, metadata={"final": True, "loss": avg_loss})
    print(f"\n[Distill] === Training abgeschlossen. Final Loss: {avg_loss:.4f} ===")

    # 9. Schneller Inferenz-Test
    decoder.eval()
    test_text = "User: Was ist Sparsity?\nAssistant:"
    test_ids = tokenizer.encode(test_text)
    test_tensor = torch.tensor([test_ids], dtype=torch.long, device=device)
    
    # Hinweis: Da wir aktuell keine komplexe repetition_penalty in decoder.py verankert haben,
    # setzen wir temperature minimal höher und übergeben top_p/k falls dort nativ unterstützt.
    generated = decoder.generate(test_tensor, max_new_tokens=80, eos_token_id=tokenizer.eos_token_id, temperature=0.85)
    new_ids = generated[len(test_ids):]
    print(f"[Test] Prompt:  '{test_text}'")
    print(f"[Test] Antwort: '{tokenizer.decode(new_ids)}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aether-Core Distillation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Anzahl Epochen überschreiben")
    args = parser.parse_args()
    distill(args.config, args.epochs)

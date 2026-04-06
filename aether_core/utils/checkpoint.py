"""
Aether-Core Checkpoint Manager.
Speichert und lädt Modellgewichte, Tokenizer-Merges und Config.
"""
import torch
import os
import json
from typing import Dict, Any


class CheckpointManager:
    """Verwaltet das Speichern und Laden aller Aether-Core Komponenten."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        step: int,
        snc: torch.nn.Module,
        decoder: torch.nn.Module,
        compression: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        metadata: Dict[str, Any] = None,
    ):
        """Speichert alle Modell-Komponenten als einzelnen Checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"aether_step_{step}.pt")

        state = {
            "step": step,
            "snc_state": snc.state_dict(),
            "decoder_state": decoder.state_dict(),
            "compression_state": compression.state_dict(),
            "metadata": metadata or {},
        }

        if optimizer is not None:
            state["optimizer_state"] = optimizer.state_dict()

        torch.save(state, path)
        print(f"[Checkpoint] Gespeichert: {path} (Step {step})")
        
        # Cleanup: Behalte maximal die 2 aktuellsten Checkpoints
        try:
            files = [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
            if len(files) > 2:
                # Nach Erstellungsdatum sortieren und die ältesten löschen
                files.sort(key=os.path.getmtime)
                for old_file in files[:-2]:
                    os.remove(old_file)
        except Exception as e:
            print(f"[Checkpoint] Warnung: Konnte alte Checkpoints nicht aufräumen: {e}")
            
        return path

    def load(
        self,
        path: str,
        snc: torch.nn.Module,
        decoder: torch.nn.Module,
        compression: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
    ) -> int:
        """Lädt einen Checkpoint und gibt den Step zurück."""
        state = torch.load(path, map_location="cpu", weights_only=False)

        snc.load_state_dict(state["snc_state"])
        decoder.load_state_dict(state["decoder_state"])
        compression.load_state_dict(state["compression_state"])

        if optimizer is not None and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])

        step = state.get("step", 0)
        print(f"[Checkpoint] Geladen: {path} (Step {step})")
        return step

    def find_latest(self) -> str:
        """Findet den neuesten Checkpoint im Verzeichnis."""
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        if not files:
            return None
        files.sort(key=lambda f: int(f.split("_")[-1].replace(".pt", "")))
        return os.path.join(self.checkpoint_dir, files[-1])

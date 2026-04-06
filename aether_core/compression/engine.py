import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class CompressionEngine(nn.Module):
    """
    Die Kompressions-Engine zur VRAM-Optimierung.
    Wandelt hochdimensionale Zustände (768) in kompakte latente Vektoren (128) um.
    """
    def __init__(self, input_dim: int = 768, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: Komprimiert d_model auf latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # Decoder: Rekonstruiert latent_dim zurück zu d_model
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Komprimiert den latenten Zustand: [batch, d_model] -> [batch, latent_dim]"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Dekomprimiert den latenten Zustand: [batch, latent_dim] -> [batch, d_model]"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoencoder Forward-Pass für das Training (Rekonstruktion)."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def get_reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Berechnet den Rekonstruktionsfehler (MSE)."""
        return F.mse_loss(x, x_recon)

if __name__ == "__main__":
    # Schneller Benchmark-Test
    engine = CompressionEngine(768, 128)
    simulated_state = torch.randn(1, 10, 768) # [batch, tokens, d_model]
    
    # Test: Ein Token komprimieren und dekomprimieren
    z = engine.encode(simulated_state)
    recon = engine.decode(z)
    
    print(f"Original-Shape: {simulated_state.shape}")
    print(f"Compressed-Shape: {z.shape} (Reduktion um den Faktor 6)")
    print(f"Reconstructed-Shape: {recon.shape} - Test erfolgreich!")

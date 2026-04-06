import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class Expert(nn.Module):
    """
    Ein einzelner 'Experte' im Mixture-of-Experts Modell.
    Besteht aus einem Feed-Forward-Netzwerk.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MoELayer(nn.Module):
    """
    Der Mixture-of-Experts Layer.
    Routet Tokens an die Top-k Experten.
    """
    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_model * 4) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, d_model]
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # [batch * seq_len, d_model]

        # 1. Gating Logits berechnen
        gate_logits = self.gate(x_flat) # [batch * seq_len, n_experts]
        
        # 2. Top-k Experten auswählen
        weights, indices = torch.topk(torch.softmax(gate_logits, dim=-1), self.top_k, dim=-1)
        # weights: [total_tokens, top_k], indices: [total_tokens, top_k]

        # 3. Output aggregieren (Sparse Activation)
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            # Für jedes Token den i-ten Experten berechnen
            exp_indices = indices[:, i]
            exp_weights = weights[:, i].unsqueeze(1)
            
            for j in range(self.n_experts):
                mask = (exp_indices == j)
                if mask.any():
                    output[mask] += exp_weights[mask] * self.experts[j](x_flat[mask])
                    
        return output.view(batch, seq_len, d_model), gate_logits

class SparseCore(nn.Module):
    """
    Das neuronale Herzstück der Aether-Core KI.
    Hier werden Tokens + symbolischer Kontext verarbeitet.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_experts: int, top_k: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MoELayer(d_model, n_experts, top_k) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, symbolic_emb: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, seq_len], symbolic_emb: [batch, d_model]
        x = self.embeddings(tokens)
        
        # Symbolischen Kontext als Prefix-Signal addieren
        x = x + symbolic_emb.unsqueeze(1)
        
        for layer in self.layers:
            x, _ = layer(x) # ignore aux_loss for now
            
        return self.ln(x)

if __name__ == "__main__":
    # Schneller Benchmark-Test
    model = SparseCore(50257, 768, 4, 8, 2)
    t = torch.randint(0, 50000, (1, 32))
    s = torch.randn(1, 768)
    out = model(t, s)
    print(f"Sparse-Core Output-Shape: {out.shape} - Test erfolgreich!")

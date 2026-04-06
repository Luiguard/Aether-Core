import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention mit kausaler Maske."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        # Kausale Maske (unteres Dreieck): verhindert Zugriff auf zukünftige Tokens
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class DecoderBlock(nn.Module):
    """Ein Transformer-Decoder-Block: Attention + FFN mit Residualverbindungen."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ChatDecoder(nn.Module):
    """
    Autoregressiver Chat-Decoder für Aether-Core.
    Generiert Token für Token eine vollständige Antwort.
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int = 4, n_heads: int = 8, max_seq_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Gewichts-Sharing: Embedding <-> LM-Head (reduziert Parameter)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, token_ids: torch.Tensor, context_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        token_ids: [batch, seq_len] - bisherige Token-Sequenz
        context_emb: [batch, d_model] - optionaler symbolischer Kontext
        Rückgabe: logits [batch, seq_len, vocab_size]
        """
        B, T = token_ids.shape
        device = token_ids.device

        pos = torch.arange(0, T, device=device).unsqueeze(0)  # [1, T]
        x = self.token_emb(token_ids) + self.pos_emb(pos)

        # Symbolischen Kontext als globalen Bias auf alle Positionen addieren
        if context_emb is not None:
            x = x + context_emb.unsqueeze(1)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        return logits

    def apply_constraints(self, logits: torch.Tensor, forbidden_tokens: List[int]) -> torch.Tensor:
        """Innovation 9.2: Constraint-Guided Search. Blockiert verbotene Tokens."""
        if forbidden_tokens:
            logits[:, forbidden_tokens] = float("-inf")
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        eos_token_id: int = 2,
        context_emb: Optional[torch.Tensor] = None,
        forbidden_tokens: Optional[List[int]] = None,
        repetition_penalty: float = 1.2,
    ) -> List[int]:
        """
        Autoregressive Generierung: Token für Token.
        prompt_ids: [1, prompt_len]
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Begrenzung auf max_seq_len
            input_ids = generated[:, -self.max_seq_len :]

            logits = self.forward(input_ids, context_emb)
            logits = logits[:, -1, :]  # Nur letztes Token

            # Constraints
            if forbidden_tokens:
                logits = self.apply_constraints(logits, forbidden_tokens)

            # Repetition Penalty
            if repetition_penalty > 1.0:
                for token_id in set(generated[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k Filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return generated[0].tolist()


if __name__ == "__main__":
    # Schneller Smoke-Test
    decoder = ChatDecoder(vocab_size=512, d_model=256, n_layers=2, n_heads=4)
    prompt = torch.randint(0, 512, (1, 5))
    logits = decoder(prompt)
    print(f"Forward OK. Logits-Shape: {logits.shape}")

    generated = decoder.generate(prompt, max_new_tokens=10)
    print(f"Generate OK. Generierte Sequenz: {generated}")

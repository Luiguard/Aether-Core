"""
Aether-Core Custom BPE Tokenizer.
Eigenentwicklung - Keine externen Abhängigkeiten.
Basiert auf Byte-Pair Encoding (UTF-8 Byte-Level).

Funktionsweise:
1. Basis-Vokabular: 256 UTF-8 Bytes (jedes Byte ist ein Token).
2. BPE-Merges: Häufige Byte-Paare werden zu neuen Tokens verschmolzen.
3. Spezial-Tokens: <|pad|>, <|bos|>, <|eos|>, <|unk|>.

Kann auf eigenen Daten trainiert werden (train_bpe).
"""
import json
import os
import re
from typing import List, Dict, Tuple, Optional


# Spezial-Token IDs (fest reserviert)
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SPECIAL_TOKENS = {
    "<|pad|>": PAD_TOKEN,
    "<|bos|>": BOS_TOKEN,
    "<|eos|>": EOS_TOKEN,
    "<|unk|>": UNK_TOKEN,
}
NUM_SPECIAL = len(SPECIAL_TOKENS)
# Byte-Tokens: IDs 4..259 (256 Bytes)
BYTE_OFFSET = NUM_SPECIAL


class AetherTokenizer:
    """
    Selbstgebauter Byte-Level BPE Tokenizer für Aether-Core.
    """

    def __init__(self, merges_path: Optional[str] = None):
        # Basis-Vokabular: Spezial-Tokens + 256 Bytes
        self.byte_to_id: Dict[int, int] = {b: b + BYTE_OFFSET for b in range(256)}
        self.id_to_byte: Dict[int, int] = {v: k for k, v in self.byte_to_id.items()}

        # BPE Merge-Tabelle: [(token_a, token_b)] -> merged_token_id
        self.merges: List[Tuple[int, int]] = []
        self.merge_to_id: Dict[Tuple[int, int], int] = {}

        # Inverses Vokabular für Decoding
        self.id_to_bytes: Dict[int, bytes] = {}
        for b in range(256):
            self.id_to_bytes[b + BYTE_OFFSET] = bytes([b])

        # Spezial-Tokens
        self.pad_token_id = PAD_TOKEN
        self.bos_token_id = BOS_TOKEN
        self.eos_token_id = EOS_TOKEN
        self.unk_token_id = UNK_TOKEN

        self.vocab_size = NUM_SPECIAL + 256  # Startwert

        if merges_path and os.path.exists(merges_path):
            self.load_merges(merges_path)

    @property
    def next_id(self) -> int:
        return self.vocab_size

    # --- Training ---
    def train_bpe(self, text: str, num_merges: int = 5000):
        """
        Trainiert BPE-Merges auf einem gegebenen Text-Korpus.
        Jeder Merge reduziert die Sequenzlänge und vergrößert das Vokabular.
        """
        print(f"[Tokenizer] Starte BPE-Training mit {num_merges} Merges...")

        # 1. Text in UTF-8 Bytes umwandeln -> Liste von Token-IDs
        tokens = [self.byte_to_id[b] for b in text.encode("utf-8")]

        for i in range(num_merges):
            # 2. Häufigstes Paar finden
            pair_counts: Dict[Tuple[int, int], int] = {}
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  # Keine sinnvollen Merges mehr

            # 3. Neues Token erstellen
            new_id = self.next_id
            self.merges.append(best_pair)
            self.merge_to_id[best_pair] = new_id

            # Bytes für das neue Token zusammensetzen
            a_bytes = self.id_to_bytes.get(best_pair[0], b"")
            b_bytes = self.id_to_bytes.get(best_pair[1], b"")
            self.id_to_bytes[new_id] = a_bytes + b_bytes

            self.vocab_size += 1

            # 4. Merges im Token-Stream anwenden
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == best_pair:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

            if (i + 1) % 1000 == 0:
                print(f"[Tokenizer] {i + 1}/{num_merges} Merges abgeschlossen. Vocab: {self.vocab_size}")

        print(f"[Tokenizer] BPE-Training fertig. Finales Vokabular: {self.vocab_size} Tokens.")

    # --- Encoding ---
    def encode(self, text: str) -> List[int]:
        """Text -> Token-IDs (mit gelernten BPE-Merges)."""
        tokens = [self.byte_to_id[b] for b in text.encode("utf-8")]

        # BPE-Merges in Reihenfolge anwenden (Priorität = Reihenfolge des Trainings)
        for pair, merged_id in self.merge_to_id.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    # --- Decoding ---
    def decode(self, token_ids: List[int]) -> str:
        """Token-IDs -> Text."""
        raw_bytes = b""
        for tid in token_ids:
            if tid in SPECIAL_TOKENS.values():
                continue  # Spezial-Tokens überspringen
            if tid in self.id_to_bytes:
                raw_bytes += self.id_to_bytes[tid]
            # Unbekannte IDs werden ignoriert

        return raw_bytes.decode("utf-8", errors="replace")

    # --- Persistenz ---
    def save_merges(self, path: str):
        """Speichert die gelernten BPE-Merges als JSON."""
        data = {
            "merges": self.merges,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"[Tokenizer] Merges gespeichert nach: {path}")

    def load_merges(self, path: str):
        """Lädt BPE-Merges aus einer JSON-Datei."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.merges = [tuple(m) for m in data["merges"]]
        self.vocab_size = data["vocab_size"]

        # Merge-Tabelle und Byte-Mapping rekonstruieren
        next_id = NUM_SPECIAL + 256
        for pair in self.merges:
            self.merge_to_id[pair] = next_id
            a_bytes = self.id_to_bytes.get(pair[0], b"")
            b_bytes = self.id_to_bytes.get(pair[1], b"")
            self.id_to_bytes[next_id] = a_bytes + b_bytes
            next_id += 1

        print(f"[Tokenizer] Merges geladen. Vokabular: {self.vocab_size} Tokens.")


if __name__ == "__main__":
    tok = AetherTokenizer()

    # Demo: Encode/Decode ohne Merges (rein Byte-Level)
    text = "Was ist ein Mixture-of-Experts Modell?"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Original:  '{text}'")
    print(f"Token-IDs: {ids[:20]}... (Laenge: {len(ids)})")
    print(f"Decoded:   '{decoded}'")

    # Demo: BPE-Training auf einem kleinen Beispieltext
    corpus = text * 50  # Kleiner Wiederholungstext zum Testen
    tok.train_bpe(corpus, num_merges=100)

    ids_after = tok.encode(text)
    print(f"\nNach BPE-Training:")
    print(f"Token-IDs: {ids_after[:20]}... (Laenge: {len(ids_after)})")
    print(f"Decoded:   '{tok.decode(ids_after)}'")
    print(f"Kompression: {len(ids)} -> {len(ids_after)} Tokens ({100 - len(ids_after)*100//len(ids)}% kuerzer)")

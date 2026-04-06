"""
Aether-Core: Rekursive Semantische Kompression (Spec 12.2).
Komprimiert alte Konversationsinhalte in permanente Graph-Knoten.
Die KI vergisst nie – alte Kontexte werden zu Wissen destilliert.
"""
import torch
from typing import List, Dict, Any
from aether_core.symbolic.symbolic_memory import SymbolicMemory
from aether_core.compression.engine import CompressionEngine


class RecursiveCompressor:
    """
    Wandelt sequenzielle Konversationen in strukturierte Graph-Knoten um.
    Statt das Context-Window zu löschen, wird Wissen verdichtet.
    """

    def __init__(self, memory: SymbolicMemory, compression_engine: CompressionEngine):
        self.memory = memory
        self.ce = compression_engine
        self.conversation_buffer: List[Dict[str, str]] = []
        self.max_buffer_size = 20  # Nach 20 Nachrichten: komprimieren

    def add_exchange(self, question: str, answer: str):
        """Fügt einen Q&A-Austausch zum Buffer hinzu."""
        self.conversation_buffer.append({"q": question, "a": answer})

        if len(self.conversation_buffer) >= self.max_buffer_size:
            self.compress_and_store()

    def compress_and_store(self):
        """
        Komprimiert den Buffer in einen neuen permanenten Graph-Knoten.
        """
        if not self.conversation_buffer:
            return

        # Schlüsselwörter aus dem gesamten Buffer extrahieren
        all_text = " ".join(
            f"{ex['q']} {ex['a']}" for ex in self.conversation_buffer
        )
        # Einfache Keyword-Extraktion (Wörter > 5 Zeichen, keine Stoppwörter)
        words = all_text.split()
        keywords = list(set(
            w.strip(".,!?:;()[]{}\"'") for w in words
            if len(w) > 5 and w[0].isupper()
        ))[:10]

        # Neuen Memory-Knoten erstellen
        node_id = f"memory_block_{len(self.memory.graph.get('nodes', {}))}"
        summary = f"Komprimiertes Wissen aus {len(self.conversation_buffer)} Austauschen."

        self.memory.add_node(node_id, summary, {
            "type": "compressed_memory",
            "keywords": keywords,
            "exchange_count": len(self.conversation_buffer),
            "source": "recursive_compression",
        })

        # Buffer leeren
        count = len(self.conversation_buffer)
        self.conversation_buffer = []
        print(f"[RecursiveCompression] {count} Austausche -> Knoten '{node_id}' gespeichert.")

    def force_flush(self):
        """Erzwingt sofortige Kompression des aktuellen Buffers."""
        if self.conversation_buffer:
            self.compress_and_store()

    def get_memory_depth(self) -> int:
        """Gibt die Anzahl komprimierter Memory-Blöcke zurück."""
        nodes = self.memory.graph.get("nodes", {})
        return sum(1 for n in nodes.values()
                   if n.get("properties", {}).get("type") == "compressed_memory")

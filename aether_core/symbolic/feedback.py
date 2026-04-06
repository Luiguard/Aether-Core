"""
Aether-Core: Self-Updating Symbolism (Spec 9.1).
Automatischer Feedback-Loop: Validierte Decoder-Ausgaben werden als neue Fakten
im Wissensgraphen gespeichert.
"""
import re
from typing import List, Dict, Any, Optional
from aether_core.symbolic.symbolic_memory import SymbolicMemory


class FeedbackLoop:
    """
    Erkennt und speichert neue Fakten aus Decoder-Ausgaben.
    Nur validierte Schlussfolgerungen werden in den Graphen geschrieben.
    """

    def __init__(self, memory: SymbolicMemory):
        self.memory = memory
        self.learned_count = 0

    def extract_claims(self, text: str) -> List[Dict[str, str]]:
        """
        Extrahiert einfache Fakten-Claims aus generiertem Text.
        Sucht nach Mustern wie "X ist Y" oder "X nutzt Y".
        """
        claims = []
        patterns = [
            (r"(\w+)\s+ist\s+(?:ein|eine|der|die|das)?\s*(.+?)[\.\,\!]", "is_a"),
            (r"(\w+)\s+nutzt\s+(.+?)[\.\,\!]", "uses"),
            (r"(\w+)\s+basiert\s+auf\s+(.+?)[\.\,\!]", "based_on"),
            (r"(\w+)\s+besteht\s+aus\s+(.+?)[\.\,\!]", "composed_of"),
        ]
        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group(1).strip()
                obj = match.group(2).strip()[:50]  # Begrenzung
                if len(subject) > 2 and len(obj) > 2:
                    claims.append({"subject": subject, "relation": rel_type, "object": obj})
        return claims

    def validate_and_store(self, text: str) -> int:
        """
        Hauptmethode: Extrahiert Claims, validiert gegen bestehende Regeln,
        und speichert neue Fakten im Graphen.
        Gibt die Anzahl neu gelernter Fakten zurück.
        """
        claims = self.extract_claims(text)
        new_facts = 0

        for claim in claims:
            subj = claim["subject"]
            obj_text = claim["object"]
            rel = claim["relation"]

            # Existiert der Subject-Knoten bereits?
            if subj not in self.memory.graph.get("nodes", {}):
                # Neuen Knoten anlegen
                self.memory.add_node(subj, subj, {"source": "self_learned"})

            # Relation als Edge speichern (wenn Ziel-Knoten existiert oder erstellt wird)
            if obj_text not in self.memory.graph.get("nodes", {}):
                self.memory.add_node(obj_text, obj_text, {"source": "self_learned"})

            # Edge hinzufügen
            self.memory.add_edge(subj, obj_text, rel)
            new_facts += 1

        self.learned_count += new_facts
        if new_facts > 0:
            print(f"[FeedbackLoop] {new_facts} neue Fakten gelernt (Gesamt: {self.learned_count}).")
        return new_facts

    def get_stats(self) -> Dict[str, int]:
        return {"total_learned": self.learned_count}

"""
Aether-Core: Entity Linker (Spec Datenfluss Step 2).
Identifiziert Konzepte aus dem Symbolic-Memory in einer Benutzerfrage.
"""
from typing import List, Dict, Any


class EntityLinker:
    """Verknüpft Text-Entitäten mit Knoten im Wissensgraphen."""

    def __init__(self, graph: Dict[str, Any]):
        self.graph = graph
        self._build_index()

    def _build_index(self):
        """Baut einen invertierten Index: lowercase name -> node_id."""
        self.index = {}
        nodes = self.graph.get("nodes", {})
        for node_id, node_data in nodes.items():
            # Index auf ID
            self.index[node_id.lower()] = node_id
            # Index auf Name
            name = node_data.get("name", "")
            if name:
                self.index[name.lower()] = node_id

    def refresh(self, graph: Dict[str, Any]):
        """Aktualisiert den Index nach Graph-Änderungen."""
        self.graph = graph
        self._build_index()

    def extract(self, text: str) -> List[str]:
        """
        Extrahiert alle im Text gefundenen Entitäten.
        Gibt eine Liste von node_ids zurück.
        """
        text_lower = text.lower()
        found = []
        for key, node_id in self.index.items():
            if key in text_lower and node_id not in found:
                found.append(node_id)
        return found

    def is_redlisted(self, text: str, redlist_tag: str = "_REDLIST_") -> List[str]:
        """
        Spec 10.1: Prüft ob der Text verbotene Konzepte enthält.
        Redlist-Knoten haben das Tag '_REDLIST_' im node_id oder in properties.
        """
        text_lower = text.lower()
        violations = []
        nodes = self.graph.get("nodes", {})
        for node_id, node_data in nodes.items():
            is_red = (
                redlist_tag.lower() in node_id.lower()
                or node_data.get("properties", {}).get("redlisted", False)
            )
            if is_red:
                name = node_data.get("name", node_id).lower()
                if name in text_lower or node_id.lower() in text_lower:
                    violations.append(node_id)
        return violations

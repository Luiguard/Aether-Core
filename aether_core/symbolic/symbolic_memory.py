import torch
import json
import os
from typing import List, Dict, Any, Optional

class SymbolicMemory:
    """
    Das Wissensmodul der Aether-Core Hybrid-KI.
    Erweitert um API-basiertes Lernen (Immediate Knowledge Acquisition).
    """
    def __init__(self, graph_path: Optional[str] = None):
        self.graph_path = graph_path
        self.graph: Dict[str, Any] = {"nodes": {}, "rules": []}
        
        if graph_path and os.path.exists(graph_path):
            self.load_graph(graph_path)
        elif graph_path:
            self.save_graph() # Initial leer anlegen

    def load_graph(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.graph["nodes"] = data.get("nodes", {})
            self.graph["rules"] = data.get("rules", [])
        print(f"[SymbolicMemory] Graph geladen: {len(self.graph['nodes'])} Knoten.")

    def save_graph(self):
        """Persistiert den aktuellen Stand in die JSON-Datei."""
        if self.graph_path:
            with open(self.graph_path, 'w', encoding='utf-8') as f:
                json.dump(self.graph, f, indent=2, ensure_ascii=False)

    # API-Learning Methoden
    def add_node(self, node_id: str, name: str, properties: Dict[str, Any]) -> bool:
        """Legt einen neuen Wissensknoten an."""
        if node_id in self.graph["nodes"]:
            return False # Konflikt: Existiert bereits
        self.graph["nodes"][node_id] = {
            "name": name,
            "properties": properties,
            "relations": []
        }
        self.save_graph()
        return True

    def add_edge(self, source_id: str, target_id: str, relation_type: str) -> bool:
        """Legt eine Relation zwischen zwei Knoten an."""
        if source_id not in self.graph["nodes"] or target_id not in self.graph["nodes"]:
            return False # Einer der Knoten fehlt
        self.graph["nodes"][source_id]["relations"].append({
            "target": target_id,
            "type": relation_type
        })
        self.save_graph()
        return True

    def add_fact(self, node_id: str, key: str, value: Any) -> bool:
        """Speichert einen Fakt als Attribut eines Knotens."""
        if node_id not in self.graph["nodes"]:
            return False
        self.graph["nodes"][node_id]["properties"][key] = value
        self.save_graph()
        return True

    def add_rule(self, rule_id: str, rule_type: str, details: Dict[str, Any]) -> bool:
        """Fügt eine neue logische Regel hinzu."""
        # Validierung: Keine doppelten Rule-IDs
        if any(r['id'] == rule_id for r in self.graph["rules"]):
            return False
        new_rule = {"id": rule_id, "type": rule_type, "details": details}
        self.graph["rules"].append(new_rule)
        self.save_graph()
        return True

    def query_concept(self, concept_id: str) -> Dict[str, Any]:
        return self.graph["nodes"].get(concept_id, {})

    def get_context_for_question(self, entities: List[str], embedding_dim: int = 768) -> torch.Tensor:
        context_vector = torch.zeros((1, embedding_dim))
        found = False
        for entity in entities:
            if entity in self.graph["nodes"]:
                found = True
                # Dummy-Signalisierung für Inferenz
                context_vector += 0.1
        return context_vector

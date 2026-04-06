import os
import json
from typing import List, Dict

class AetherIngest:
    """
    Das Tool zum automatischen Wissensaufbau (Phase 5).
    Verarbeitet Dokumente und bereitet sie für das Symbolic-Memory vor.
    """
    def __init__(self, target_graph: str):
        self.target_graph = target_graph

    def process_file(self, file_path: str) -> List[Dict]:
        """
        Extrahiert Entitäten und Relationen aus einer Datei (Placeholder-Logik).
        In einer vollen Version würde hier ein LLM oder NLP-Parser (Spacy/NLTK) laufen.
        """
        ext = os.path.splitext(file_path)[1].lower()
        print(f"[Ingest] Verarbeite Datei: {file_path} ({ext})")
        
        # Simulierter Extraktionsprozess
        found_nodes = [
            {"id": "Example_" + os.path.basename(file_path), "name": "Extrahiertes Wissen", "type": "document"}
        ]
        return found_nodes

    def update_knowledge_graph(self, nodes: List[Dict]):
        """Schreibt neue Knoten in den bestehenden Wissensgraphen."""
        try:
            with open(self.target_graph, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            for node in nodes:
                node_id = node["id"]
                graph_data["nodes"][node_id] = node
                
            with open(self.target_graph, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            print(f"[Ingest] Graph erfolgreich aktualisiert: +{len(nodes)} Knoten.")
        except Exception as e:
            print(f"[Ingest] Fehler beim Aktualisieren des Graphen: {e}")

if __name__ == "__main__":
    # Testlauf
    ingest = AetherIngest("aether_core/data/ki_architektur.json")
    nodes = ingest.process_file("demo_dokument.txt")
    ingest.update_knowledge_graph(nodes)

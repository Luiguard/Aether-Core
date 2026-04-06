"""
Aether-Core: Knowledge Integrator (DeepSeek-R2 to Symbolic Memory).
Extrahiert strukturiertes Wissen über die API und iteriert es direkt in den Graphen.
"""
import json
import time
import hashlib
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ValidationError

class FactItem(BaseModel):
    node: str
    key: str
    value: Any

class EdgeItem(BaseModel):
    source: str
    relation: str
    target: str

class NodeItem(BaseModel):
    id: str
    label: str
    attributes: Dict[str, Any]

class RuleItem(BaseModel):
    if_cond: str
    then_action: str

class ConstraintItem(BaseModel):
    condition: str
    action: str

class ExtractedKnowledge(BaseModel):
    nodes: List[NodeItem] = []
    edges: List[EdgeItem] = []
    facts: List[FactItem] = []
    rules: List[RuleItem] = []
    constraints: List[ConstraintItem] = []


class DeepSeekIntegrator:
    """Verbindet DeepSeek API mit dem Symbolic Memory."""
    
    def __init__(self, teacher_client, memory_api_url: str = "http://127.0.0.1:8444"):
        self.teacher = teacher_client
        self.memory_api_url = memory_api_url
        import requests
        self.req = requests # lazy load
        
    def generate_query(self, topic: str) -> str:
        """Erzeugt eine strikte Query, die reinen JSON-Output anfordert."""
        return f"""Du bist ein struktureller Wissensextraktor. Analysiere das Thema '{topic}'.
Gib AUSSCHLIESSLICH gültiges JSON zurück. KEINEN erklärenden Text.
Dein Return-Format MUSS exakt dieses Schema einhalten:
{{
  "nodes": [{{"id": "eindeutige_id", "label": "Anzeigename", "attributes": {{"kategorie": "..."}}}}],
  "edges": [{{"source": "id1", "relation": "beziehung", "target": "id2"}}],
  "facts": [{{"node": "id1", "key": "faktname", "value": "wert"}}],
  "rules": [{{"if_cond": "Wenn X passiert", "then_action": "Dann folgere Y"}}],
  "constraints": [{{"condition": "Bedingung Z", "action": "Blockiere/Erlaube"}}]
}}

Achte unbedingt darauf, dass keine Strings innerhalb des JSONs nicht geschlossene Quotes, Tabs, oder Zeilenumbrüche (außer escaped \\n) enthalten, damit es von Python json.loads geparst werden kann.
"""
        
    def _parse_and_validate(self, raw_response: str) -> Optional[ExtractedKnowledge]:
        """Tries to extract JSON and validate it against the Pydantic schema."""
        try:
            # Versuche Markdown-JSON-Blöcke zu parsen
            if "```json" in raw_response:
                start = raw_response.find("```json") + 7
                end = raw_response.rfind("```")
            else:
                start = raw_response.find("{")
                end = raw_response.rfind("}") + 1
            
            if start >= 0 and end > start:
                json_str = raw_response[start:end].strip()
                data = json.loads(json_str)
                return ExtractedKnowledge(**data)
            
            print("[Integrator] Konnte keine gültigen JSON-Grenzen finden.")
            return None
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[Integrator] Validation Error: {e}")
            # Bei Längen-Überschreitungen oder unvollständigem JSON durch max_tokens Limit
            print(f"[Integrator] Raw Output Snippet: {raw_response[:200]} ... {raw_response[-200:]}")
            return None

    def acquire_topic(self, topic: str) -> bool:
        """Führt den kompletten Flow: Query -> Validate -> Store aus."""
        print(f"[Integrator] Starte Akquise für Thema: '{topic}'")
        
        # 1. Query an DeepSeek
        messages = [
            {"role": "system", "content": "Verhalte dich wie eine reine Daten-Extraktions-API. Keine Konversation."},
            {"role": "user", "content": self.generate_query(topic)}
        ]
        
        raw_response = self.teacher._call(messages, temperature=0.1, max_tokens=4000)
        if not raw_response:
             print("[Integrator] Keine Antwort von API erhalten.")
             return False

        # 2. Validation
        knowledge = self._parse_and_validate(raw_response)
        if not knowledge:
             print("[Integrator] Fehler bei Extraktion/Validierung.")
             return False
             
        # 3. Store in Memory via REST API
        print(f"[Integrator] Validiert: {len(knowledge.nodes)} Nodes, {len(knowledge.edges)} Edges.")
        self._write_to_memory(knowledge)
        return True
        
    def _write_to_memory(self, k: ExtractedKnowledge):
        """Schreibt validierte Items in die lokale Aether-Core API."""
        timestamp = int(time.time())
        
        # Helper zum Senden
        def _post(endpoint: str, payload: dict):
            try:
                res = self.req.post(f"{self.memory_api_url}/{endpoint}", json=payload)
                if res.status_code == 200:
                    return True
                elif res.status_code == 400 and "Konflikt" in res.text:
                    # Knoten existiert bereits, völlig ok im autonomen Betrieb
                    return True
                else:
                    print(f"WARN: API {endpoint} ({res.status_code}) - {res.text}")
                    return False
            except Exception as e:
                print(f"ERROR: API Connection {e}")
                return False

        # 1. Nodes
        for n in k.nodes:
            attrs = n.attributes.copy()
            attrs["_source"] = "DeepSeek-Teacher"
            attrs["_ts"] = timestamp
            attrs["_hash"] = hashlib.md5(f"{n.id}{timestamp}".encode()).hexdigest()
            _post("node", {"id": n.id, "name": n.label, "properties": attrs})
            
        # 2. Edges
        for e in k.edges:
             _post("edge", {"source_id": e.source, "target_id": e.target, "relation_type": e.relation})
             
        # 3. Facts
        for f in k.facts:
             _post("fact", {"id": f.node, "key": f.key, "value": f.value})
             
        # Regeln blockieren wir erstmal (oder rufen den passenden Endpunkt auf)
        for r in k.rules:
             rule_id = hashlib.md5(f"{r.if_cond}{r.then_action}".encode()).hexdigest()
             _post("rule", {"id": f"rule_{rule_id[:8]}", "type": "logic", "details": {"if": r.if_cond, "then": r.then_action}})
             
        print("[Integrator] Schreibvorgang in Graph abgeschlossen.")

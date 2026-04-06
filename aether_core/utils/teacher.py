"""
Aether-Core: DeepSeek Teacher Client.
Nutzt die DeepSeek API als Teacher-Modell für Distillation und Wissensextraktion.
Key wird ausschließlich aus Umgebungsvariable gelesen.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional


class TeacherClient:
    """
    Client für die DeepSeek API (OpenAI-kompatibel).
    Dient als Teacher für Knowledge Distillation und als Wissensquelle.
    """

    def __init__(self):
        # 1. Erst Umgebungsvariable prüfen (fuer Noobs/Terminal)
        self.api_key = os.environ.get("AETHER_TEACHER_API_KEY", "")
        
        # 2. Falls leer, direkt in config.yaml nachsehen
        if not self.api_key:
            try:
                import yaml
                config_path = "config.yaml"
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    self.api_key = cfg.get("teacher", {}).get("api_key", "")
            except:
                pass
                
        self.api_url = os.environ.get("AETHER_TEACHER_API_URL", "https://api.deepseek.com/v1/chat/completions")
        self.model = os.environ.get("AETHER_TEACHER_MODEL", "deepseek-chat")

        if not self.api_key:
            print("[Teacher] WARNUNG: Kein API-Key gefunden. Nutze Dashboard zum Speichern oder setze AETHER_TEACHER_API_KEY.")

    def _call(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Sendet eine Anfrage an die DeepSeek API und prüft dynamisch auf Keys."""
        # Dynamisches Key-Refresh falls leer
        if not self.api_key:
            self.api_key = os.environ.get("AETHER_TEACHER_API_KEY", "")
            if not self.api_key:
                try:
                    import yaml
                    if os.path.exists("config.yaml"):
                        with open("config.yaml", "r", encoding="utf-8") as f:
                            cfg = yaml.safe_load(f)
                        self.api_key = cfg.get("teacher", {}).get("api_key", "")
                except: pass

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            r = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Teacher] API-Fehler: {e}")
            return ""

    # --- Distillation: Trainingsdaten generieren ---
    def generate_qa_pair(self, topic: str) -> List[Dict[str, str]]:
        """Generiert 5 diverse Frage-Antwort-Paare zu einem Thema."""
        messages = [
            {"role": "system", "content": "Du bist ein präziser KI-Lehrer. Erzeuge 5 unterschiedliche, natürliche Dialog-Paare."},
            {"role": "user", "content": f"Erzeuge 5 kurze Frage-Antwort-Paare (User/Assistant) zum Thema: {topic}. Gib NUR ein JSON-Array zurück: [{{'question': '...', 'answer': '...'}}]"},
        ]
        raw = self._call(messages, temperature=0.7, max_tokens=1024)
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0:
                return json.loads(raw[start:end])
        except:
            pass
        return [{"question": topic, "answer": "Fehler bei Generierung"}]

    def generate_training_batch(self, topics: List[str]) -> List[Dict[str, str]]:
        """Generiert einen großen Batch an Trainingsdaten."""
        print(f"[Teacher] Generiere massiven Batch für {len(topics)} Themen...")
        all_pairs = []
        for i, topic in enumerate(topics):
            pairs = self.generate_qa_pair(topic)
            all_pairs.extend(pairs)
            if (i + 1) % 5 == 0:
                print(f"[Teacher] {len(all_pairs)} Paare bisher generiert...")
        return all_pairs

    # --- Wissensextraktion: Graph automatisch befüllen ---
    def extract_knowledge(self, topic: str) -> Dict[str, Any]:
        """
        Extrahiert strukturiertes Wissen aus der DeepSeek API.
        Rückgabe: Knoten und Kanten für den Wissensgraphen.
        """
        messages = [
            {"role": "system", "content": (
                "Du bist ein Wissens-Extraktor. Gib strukturiertes Wissen als JSON zurück. "
                "Format: {\"nodes\": [{\"id\": \"...\", \"name\": \"...\", \"properties\": {}}], "
                "\"edges\": [{\"source\": \"...\", \"target\": \"...\", \"type\": \"...\"}]}. "
                "Antworte NUR mit gültigem JSON, kein Markdown."
            )},
            {"role": "user", "content": f"Extrahiere die wichtigsten Konzepte und Relationen zum Thema: {topic}"},
        ]
        raw = self._call(messages, temperature=0.1, max_tokens=512)

        try:
            # JSON aus der Antwort extrahieren
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            print(f"[Teacher] Konnte JSON nicht parsen: {raw[:100]}...")

        return {"nodes": [], "edges": []}

    def save_training_data(self, pairs: List[Dict[str, str]], path: str = "aether_core/data/training_data.json"):
        """Speichert generierte Trainingsdaten als JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"[Teacher] Trainingsdaten gespeichert: {path} ({len(pairs)} Paare)")


if __name__ == "__main__":
    client = TeacherClient()

    if not client.api_key:
        print("Setze den Key: $env:AETHER_TEACHER_API_KEY = 'dein-key'")
    else:
        # Test: Ein einzelnes QA-Paar generieren
        pair = client.generate_qa_pair("Was ist ein Mixture-of-Experts Modell?")
        print(f"Frage:   {pair['question']}")
        print(f"Antwort: {pair['answer'][:200]}...")

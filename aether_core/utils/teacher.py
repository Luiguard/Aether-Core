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
        self.api_key = os.environ.get("AETHER_TEACHER_API_KEY", "")
        self.api_url = os.environ.get("AETHER_TEACHER_API_URL", "https://api.deepseek.com/v1/chat/completions")
        self.model = os.environ.get("AETHER_TEACHER_MODEL", "deepseek-chat")

        if not self.api_key:
            print("[Teacher] WARNUNG: Kein API-Key gesetzt. Setze AETHER_TEACHER_API_KEY als Umgebungsvariable.")

    def _call(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Sendet eine Anfrage an die DeepSeek API."""
        current_key = os.environ.get("AETHER_TEACHER_API_KEY", self.api_key)
        headers = {
            "Authorization": f"Bearer {current_key}",
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
    def generate_qa_pair(self, topic: str) -> Dict[str, str]:
        """Generiert ein Frage-Antwort-Paar zu einem Thema (für Student-Training)."""
        messages = [
            {"role": "system", "content": "Du bist ein präziser KI-Lehrer. Antworte kurz und faktenbasiert auf Deutsch."},
            {"role": "user", "content": f"Erkläre kurz und präzise: {topic}"},
        ]
        answer = self._call(messages, temperature=0.3, max_tokens=256)
        return {"question": topic, "answer": answer}

    def generate_training_batch(self, topics: List[str]) -> List[Dict[str, str]]:
        """Generiert einen ganzen Batch an Trainingsdaten."""
        print(f"[Teacher] Generiere {len(topics)} Trainingsdaten-Paare...")
        pairs = []
        for i, topic in enumerate(topics):
            pair = self.generate_qa_pair(topic)
            if pair["answer"]:
                pairs.append(pair)
            if (i + 1) % 10 == 0:
                print(f"[Teacher] {i + 1}/{len(topics)} Paare generiert.")
        print(f"[Teacher] Fertig. {len(pairs)} gueltige Paare.")
        return pairs

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

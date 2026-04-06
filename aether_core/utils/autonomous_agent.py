"""
Aether-Core: Autonomous Knowledge Agent.
Durchläuft den Graphen, erkennt Wissenslücken und fragt selbstständig DeepSeek (Teacher) ab, 
um den Graph in Echtzeit über die API zu füllen.
"""
import time
import random
import requests
from typing import List, Dict, Any, Optional

from aether_core.utils.teacher import TeacherClient
from aether_core.utils.integrator import DeepSeekIntegrator


class KnowledgeGapDetector:
    """Bestimmt, worüber das System als Nächstes Wissen sammeln soll."""
    def __init__(self, api_url: str = "http://127.0.0.1:8444"):
        self.api_url = api_url
        
    def fetch_graph(self) -> Dict[str, Any]:
        """Holt aktuellen Graph der REST API."""
        try:
            r = requests.get(f"{self.api_url}/graph")
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return {"nodes": {}, "rules": []}

    def detect_next_gap(self) -> str:
        """Findet das 'unterentwickeltste' Konzept."""
        graph = self.fetch_graph()
        nodes = graph.get("nodes", {})
        
        if not nodes:
            # Fallback wenn der Graph komplett leer ist
            start_concepts = ["Neuronale Netze", "Künstliche Intelligenz", "Quantenphysik", "Bioinformatik"]
            return random.choice(start_concepts)
            
        gaps = []
        for node_id, data in nodes.items():
            name = data.get("name", node_id)
            relations = len(data.get("relations", []))
            facts = len([k for k in data.get("properties", {}).keys() if not k.startswith("_")])
            
            # Formel für Vollständigkeit. Weniger ist schlechter.
            score = relations * 2 + facts
            
            # Wir suchen gezielt Knoten mit wenig Relationen/Fakten
            if score < 5:
                gaps.append((score, name))
                
        if not gaps:
            # Alle bisherigen Konzepte sind 'voll'. Wähle ein zufälliges um es zu vertiefen.
            node_id = random.choice(list(nodes.keys()))
            return nodes[node_id].get("name", node_id)
            
        # Wähle zufällig eins der am schlechtesten angebundenen Konzepte aus
        # Sortiere aufsteigend nach score, nehme die Top 3 schwächsten, und wähle zufällig
        gaps.sort(key=lambda x: x[0])
        weakest = gaps[:3]
        return random.choice(weakest)[1]


class AutonomousAgent:
    """Der autonome Scheduler & Runner."""
    def __init__(self, api_url: str = "http://127.0.0.1:8444"):
        self.api_url = api_url
        self.detector = KnowledgeGapDetector(api_url)
        self.teacher = TeacherClient()
        # Custom integrators that uses DeepSeek-R2-Autonomous tag
        self.integrator = DeepSeekIntegrator(self.teacher, api_url)
        
    def run_cycle(self) -> bool:
        """Führt einen einzelnen Lernzyklus durch."""
        print("\n" + "="*50)
        print("[AutoAgent] Starte neuen Lern-Zyklus...")
        
        # 1. Spüre Lücke auf
        target_topic = self.detector.detect_next_gap()
        print(f"[AutoAgent] Wissenslücke identifiziert: '{target_topic}'")
        
        # 2. Akquise durch Integrator anstoßen
        # Wir modifizieren den generierten Payload des Integrators minimal für Autonomie
        # Der DeepSeekIntegrator kümmert sich um alles weitere (Validator, Mapper, Output).
        success = self.integrator.acquire_topic(f"Details, Fakten, Typen und Konzepte zu {target_topic}")
        
        if success:
             print("[AutoAgent] Wissen erfolgreich in den Graph extrahiert und gelernt.")
        else:
             print("[AutoAgent] Zyklus ohne nutzbares Ergebnis beendet. Nächster Versuch später.")
             
        return success

    def run_loop(self, duration_s: int = 300, interval_s: int = 30):
        """
        Führt den Agenten periodisch aus.
        duration_s: Wie lange der Test laufen soll in Sekunden.
        interval_s: Wartezeit zwischen den Anfragen.
        """
        print(f"\n[AutoAgent] Starte automatischen Lern-Loop. Testdauer: {duration_s}s, Zyklus: {interval_s}s")
        start_time = time.time()
        
        # Warten, bis der API Server wirklich da ist
        api_ready = False
        for _ in range(10):
            try:
                requests.get(f"{self.api_url}/health")
                api_ready = True
                break
            except:
                time.sleep(1)
                
        if not api_ready:
            print("[AutoAgent] API Server nicht erreichbar. Abbruch.")
            return

        cycles = 0
        while time.time() - start_time < duration_s:
            cycles += 1
            print(f"\n[AutoAgent] --- Zyklus {cycles} / Zeit: {int(time.time() - start_time)}s ---")
            self.run_cycle()
            
            # Restzeit berechnen
            elapsed = time.time() - start_time
            if elapsed >= duration_s:
                break
                
            sleep_time = min(interval_s, duration_s - elapsed)
            print(f"[AutoAgent] Pausiere für {int(sleep_time)} Sekunden...")
            time.sleep(sleep_time)

        print("\n" + "="*50)
        print(f"[AutoAgent] Testlauf beendet. Lief für {int(time.time() - start_time)} Sekunden.")
        
        
if __name__ == "__main__":
    agent = AutonomousAgent()
    # Führe einen 300 Sekunden Test aus, lerne alle 45 Sekunden etwas neues.
    agent.run_loop(duration_s=300, interval_s=45)

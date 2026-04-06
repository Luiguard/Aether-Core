"""
Aether-Core: Autonomous Knowledge Agent.
Durchläuft den Graphen, erkennt Wissenslücken und fragt selbstständig DeepSeek (Teacher) ab, 
um den Graph in Echtzeit über die API zu füllen.
"""
import time
import random
import requests
from typing import List, Dict, Any, Optional, Set

from aether_core.utils.teacher import TeacherClient
from aether_core.utils.integrator import DeepSeekIntegrator


# Breites Seed-Vokabular für echte Wissens-Diversität
EXPLORATION_SEEDS = [
    "Reinforcement Learning", "Generative Adversarial Networks", "Bayessche Statistik",
    "Computerlinguistik", "Wissensgraphen", "Recurrent Neural Networks",
    "Convolutional Neural Networks", "Transfer Learning", "Few-Shot Learning",
    "Selbstüberwachtes Lernen", "Attention-Mechanismus", "Tokenisierung",
    "Worteinbettungen", "Sentimentanalyse", "Named Entity Recognition",
    "Sprachsynthese", "Spracherkennung", "Bildklassifikation",
    "Objekterkennung", "Semantische Segmentierung", "Anomalieerkennung",
    "Zeitreihenanalyse", "Empfehlungssysteme", "Clustering-Algorithmen",
    "Dimensionsreduktion", "Feature Engineering", "Hyperparameter-Optimierung",
    "Neuronale Architektursuche", "Modellkompression", "Wissensdestillation",
    "Föderiertes Lernen", "Differentielle Privatsphäre", "Erklärbare KI",
    "Kausale Inferenz", "Graphneuronale Netze", "Diffusionsmodelle",
    "Vision Transformer", "Multimodale KI", "Prompt Engineering",
    "Retrieval-Augmented Generation", "Kontextfenster", "Feintuning",
    "RLHF", "Chain-of-Thought Reasoning", "Agentische KI-Systeme",
    "Robotik und Steuerung", "Evolutionäre Algorithmen", "Schwarmintelligenz",
    "Quantencomputing", "Kryptographie", "Informationstheorie",
    "Spieltheorie", "Entscheidungsbäume", "Random Forests",
    "Support Vector Machines", "Kernel-Methoden", "Ensemble-Methoden",
    "Gradientenabstieg", "Adam-Optimierer", "Batch-Normalisierung",
    "Dropout-Regularisierung", "Residuale Verbindungen", "Autoencoder",
    "Variational Autoencoder", "Contrastive Learning", "BERT-Architektur",
    "GPT-Architektur", "Mixture-of-Experts", "Sparse Attention",
    "Datenaugmentation", "Aktives Lernen", "Curriculum Learning",
    "Meta-Learning", "Zero-Shot Learning", "Continual Learning",
    "Neuronale Programmierung", "Symbolische KI", "Logische Programmierung",
    "Ontologie-Engineering", "Semantisches Web", "Linked Data",
    "Netzwerkanalyse", "Soziale Netzwerktheorie", "Bioinformatik",
    "Proteinfaltung", "Arzneimittelentwicklung", "Medizinische Bildgebung",
    "Autonomes Fahren", "Drohnennavigation", "Natürliche Sprachgenerierung",
    "Dialogsysteme", "Fragenbeantwortung", "Textklassifikation",
    "Informationsextraktion", "Relation Extraction", "Coreference Resolution",
    "Maschinelles Lesen", "Dokumentenverständnis", "OCR-Technologie",
]


class KnowledgeGapDetector:
    """Bestimmt, worüber das System als Nächstes Wissen sammeln soll."""
    
    # Session-weiter Cooldown (überlebt Zyklen, aber nicht Neustarts)
    _processed_topics: Dict[str, float] = {}  # {topic_name: last_processed_timestamp}
    _explored_seeds: Set[str] = set()  # Bereits erkundete Seed-Themen
    
    COOLDOWN_SECONDS = 900  # 15 Minuten Cooldown pro Thema
    
    def __init__(self, api_url: str = "http://127.0.0.1:8444"):
        self.api_url = api_url
        
    def fetch_graph(self) -> Dict[str, Any]:
        """Holt aktuellen Graph der REST API."""
        try:
            r = requests.get(f"{self.api_url}/graph", timeout=5)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return {"nodes": {}, "rules": []}

    def _is_on_cooldown(self, topic_name: str) -> bool:
        """Prüft ob ein Thema kürzlich bereits bearbeitet wurde."""
        last_time = self._processed_topics.get(topic_name.lower(), 0)
        return (time.time() - last_time) < self.COOLDOWN_SECONDS
    
    def mark_processed(self, topic_name: str):
        """Markiert ein Thema als bearbeitet (Session-Cooldown)."""
        self._processed_topics[topic_name.lower()] = time.time()

    def detect_next_gap(self) -> str:
        """Findet das nächste zu lernende Konzept mit intelligenter Diversifikation."""
        graph = self.fetch_graph()
        nodes = graph.get("nodes", {})
        
        if not nodes:
            # Graph leer → Zufälliges Seed-Thema
            available_seeds = [s for s in EXPLORATION_SEEDS if not self._is_on_cooldown(s)]
            if available_seeds:
                return random.choice(available_seeds[:10])
            return random.choice(EXPLORATION_SEEDS)
        
        # 1. PHASE: Echte Lücken im bestehenden Graphen finden
        # Berechne Score für jeden EINZIGARTIGEN Knotennamen (dedupliziert)
        name_scores: Dict[str, Dict] = {}  # {name: {score, node_ids, ...}}
        
        # Zähle auch eingehende Kanten (bidirektional)
        incoming_count: Dict[str, int] = {}
        for nid, d in nodes.items():
            for rel in d.get("relations", []):
                target = rel.get("target", "")
                incoming_count[target] = incoming_count.get(target, 0) + 1
        
        for node_id, data in nodes.items():
            name = data.get("name", node_id)
            name_lower = name.lower()
            
            # Cooldown-Check nach NAME (nicht nach Node-ID)
            if self._is_on_cooldown(name):
                continue
            
            props = data.get("properties", {})
            outgoing = len(data.get("relations", []))
            incoming = incoming_count.get(node_id, 0)  
            facts = len([k for k in props.keys() if not k.startswith("_")])
            
            # Verbesserte Score-Formel: Outgoing + Incoming + Facts
            score = outgoing + incoming + facts
            
            if name_lower not in name_scores or score < name_scores[name_lower]["score"]:
                name_scores[name_lower] = {
                    "score": score,
                    "name": name,
                    "node_id": node_id
                }
        
        # Filtere auf echte Lücken (Score < 8 für höheren Qualitätsstandard)
        gaps = [(info["score"], info["name"]) for info in name_scores.values() if info["score"] < 8]
        
        if gaps:
            # Sortiere nach Score, wähle zufällig aus den Top 5 Schwächsten
            gaps.sort(key=lambda x: x[0])
            candidates = gaps[:min(5, len(gaps))]
            chosen = random.choice(candidates)[1]
            return chosen
        
        # 2. PHASE: Keine Lücken mehr → EXPLORIERE neues Wissen
        # Finde Seed-Themen die noch NICHT im Graph sind
        existing_names = {d.get("name", "").lower() for d in nodes.values()}
        new_seeds = [
            s for s in EXPLORATION_SEEDS 
            if s.lower() not in existing_names 
            and s.lower() not in self._explored_seeds
            and not self._is_on_cooldown(s)
        ]
        
        if new_seeds:
            chosen = random.choice(new_seeds[:10])
            self._explored_seeds.add(chosen.lower())
            return chosen
        
        # 3. PHASE: Alles bekannt → Vertiefte Exploration
        # Wähle den ältesten existierenden Node für Re-Analyse
        oldest = sorted(
            [(nid, d) for nid, d in nodes.items() if not self._is_on_cooldown(d.get("name", nid))],
            key=lambda x: x[1].get("properties", {}).get("_ts", 0)
        )
        
        if oldest:
            return oldest[0][1].get("name", oldest[0][0])
        
        # Absoluter Notfall: Warte
        return random.choice(EXPLORATION_SEEDS)


class AutonomousAgent:
    """Der autonome Scheduler & Runner."""
    def __init__(self, api_url: str = "http://127.0.0.1:8444"):
        self.api_url = api_url
        self.detector = KnowledgeGapDetector(api_url)
        self.teacher = TeacherClient()
        self.integrator = DeepSeekIntegrator(self.teacher, api_url)
        self._consecutive_failures = 0
        
    def run_cycle(self) -> bool:
        """Führt einen einzelnen Lernzyklus durch."""
        print("\n" + "="*50)
        print("[AutoAgent] Starte neuen Lern-Zyklus...")
        
        # 1. Spüre Lücke auf
        target_topic = self.detector.detect_next_gap()
        print(f"[AutoAgent] Wissenslücke identifiziert: '{target_topic}'")
        
        # 2. Sofort als bearbeitet markieren (BEVOR Akquise startet)
        #    Verhindert wiederholte Auswahl bei Fehlschlag
        self.detector.mark_processed(target_topic)
        
        # 3. Akquise durch Integrator anstoßen
        success = self.integrator.acquire_topic(f"Details, Fakten, Typen und Konzepte zu {target_topic}")
        
        if success:
            self._consecutive_failures = 0
            print(f"[AutoAgent] ✓ Wissen zu '{target_topic}' erfolgreich in den Graph integriert.")
        else:
            self._consecutive_failures += 1
            print(f"[AutoAgent] ✗ Zyklus für '{target_topic}' ohne Ergebnis (Fehler #{self._consecutive_failures}).")
            
            # Bei wiederholten Fehlern: Längere Pause
            if self._consecutive_failures >= 3:
                print("[AutoAgent] Mehrere Fehlschläge in Folge. Verlängere Pause auf 120s...")
                
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
                requests.get(f"{self.api_url}/health", timeout=3)
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
            elapsed = int(time.time() - start_time)
            print(f"\n[AutoAgent] --- Zyklus {cycles} / Zeit: {elapsed}s ---")
            self.run_cycle()
            
            # Restzeit berechnen
            elapsed = time.time() - start_time
            if elapsed >= duration_s:
                break
            
            # Bei vielen Fehlern: Längere Pause
            actual_interval = interval_s
            if self._consecutive_failures >= 3:
                actual_interval = max(interval_s, 120)
                
            sleep_time = min(actual_interval, duration_s - elapsed)
            print(f"[AutoAgent] Pausiere für {int(sleep_time)} Sekunden...")
            time.sleep(sleep_time)

        print("\n" + "="*50)
        print(f"[AutoAgent] Testlauf beendet. Lief für {int(time.time() - start_time)} Sekunden, {cycles} Zyklen.")
        

if __name__ == "__main__":
    agent = AutonomousAgent()
    # Führe einen 300 Sekunden Test aus, lerne alle 45 Sekunden etwas neues.
    agent.run_loop(duration_s=300, interval_s=45)


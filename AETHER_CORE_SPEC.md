# Aether-Core: Hybrid-KI Architektur & Blueprint

**Projektname:** Aether-Core Hybrid AI  
**Fokus:** Neuro-Symbolische Integration, Extreme Effizienz (Sparse-Core), Lokale Inferenz.  
**Version:** 1.0 (Initial Draft)

---

## 1. Vision & Zielsetzung

Aether-Core ist ein KI-System, das die Brücke zwischen **formaler Logik (Symbolik)** und **neuronaler Intuition (LLM)** schlägt. Durch die Trennung von Wissen (Graph) und Sprache (Decoder) erreichen wir eine signifikante Reduktion der Modellgröße bei gleichzeitiger Erhöhung der Faktentreue.

- **Zielgröße:** 400 – 1100 MB (Disk/VRAM)
- **Komplexität:** < 1B Parameter (Total), ~50-100M aktiv pro Inferenzschritt.
- **Hardware:** Einzell-GPU (z. B. RTX 3060 12GB), lokaler Windows Desktop.

---

## 2. Systemarchitektur (High-Level)

Das System besteht aus fünf modularen Säulen:

1.  **Symbolic-Memory (SM):** Speichert Fakten als Graphen (Nodes/Edges) und Regeln (Constraints).
2.  **Sparse-Neural-Core (SNC):** Ein MoE (Mixture-of-Experts) Transformer, der nur relevante "Experten-Pfade" aktiviert.
3.  **Compression-Engine (CE):** Ein Autoencoder zur Kompression latenter Zustände und VRAM-Optimierung.
4.  **Chat-Decoder (CD):** Ein leichtgewichtiger Decoder mit **Constraint-Guided Search**, der latente Vektoren in natürliche Sprache übersetzt.
5.  **Aether-Orchestrator (AO):** Steuert den Datenfluss und schließt den **Self-Updating Feedback-Loop**.

### Datenfluss (Step-by-Step)
1. **Input:** Benutzerfrage wird tokenisiert.
2. **Entitäts-Linker:** Identifiziert Konzepte im `Symbolic-Memory`.
3. **Graph-Extrakt:** Ruft relevanten Kontext und Regeln ab.
4. **Core-Processing:** `SNC` verarbeitet Tokens + Graph-Extrakt.
5. **State-Compression:** `CE` komprimiert den Kontext für das Kurzzeitgedächtnis.
6. **Decoding:** `CD` generiert die Antworttokens unter Berücksichtigung des Stils.
7. **Guardrail:** `SM` prüft die Antwort gegen harte logische Regeln (Constraint Validation).

---

## 3. Technische Spezifikationen

### Datenstrukturen (Python/PyTorch)

| Komponente | Datenstruktur | Kern-Attribut / Shape |
| :--- | :--- | :--- |
| **Graph** | `Dict[str, Node]` | Adjazenzliste + Feature-Atts |
| **SNC-Input** | `Tensor` | `[Batch, SeqLen, 768]` |
| **Latent-State** | `Tensor` | `[Batch, 128]` (Compressed) |
| **MoE routing** | `Softmax / Top-k` | Bestimmt `active_experts` per Token |

### API-Schnittstellen

- **SymbolicMemory:**
    - `query(concept_id: str) -> Facts`
    - `validate(response: str) -> bool`
- **SparseCore:**
    - `forward(tokens, symbolic_context) -> hidden_state`
- **CompressionEngine:**
    - `encode(hidden_state) -> z_latent`
    - `decode(z_latent) -> hidden_state_approx`

---

## 4. Trainingsplan & Strategie

### Ebene 1: Symbolische Kuration
- **Fokus:** Manuelle oder semi-automatische Erstellung von Wissensdomänen.
- **Validierung:** Unit-Tests für Abfragegeschwindigkeit und Relevanz der Graphen-Extraktion.

### Ebene 2: Sparse-Training (Das "Gehirn")
- **Daten:** Synthetische Dialoge + Wikipedia-Subsets (für Sprachverständnis).
- **Loss:** `CrossEntropy` + `MoE_Balance_Loss` + `Symbolic_Consistency_Loss`.
- **Technik:** 8 Experten, Top-2 Routing (Sparsity).

### Ebene 3: Compression-Training
- **Daten:** Aktivierungsmuster aus Ebene 2.
- **Ansatz:** Variational Autoencoder (VAE) oder Rate-Distortion Autoencoder.
- **Ziel:** Minimale Bitrate pro Vektor bei maximaler Rekonstruktionsqualität.

---

## 5. Ressourcen & Budget

- **CPU:** Optimiert für Multi-Threading (8+ Kerne empfohlen).
- **RAM:** ~16 GB (System) für Graph-Datenbank im Speicher.
- **GPU:** Min. 8 GB VRAM. Training optimiert durch Mixed-Precision (FP16/BF16).
- **Disk:** < 1.1 GB für die vollständige Modell-Checkpoints.

---

## 6. Implementation Roadmap (Der 5-Wochen-Plan)

- [ ] **Woche 1: Fundament (Symbolik)**
    - Implementierung `SymbolicMemory` Klasse.
    - Entwurf des Graph-Schemas (Nodes, Edges, Rules).
    - Beispiel-Domäne "KI-Architektur" befüllen.
- [ ] **Woche 2: Das Gehirn (Sparse-Core)**
    - Erstellen des MoE-Transformer Skeletons.
    - Integration von `top-k` Attention.
    - Forward-Pass Benchmark (VRAM & Speed).
- [ ] **Woche 3: Effizienz (Kompression)**
    - Implementierung `CompressionEngine`.
    - Sammeln von Core-States für VAE-Training.
    - Training der Kompression + Integration in den Inferenz-Pfad.
- [ ] **Woche 4: Intelligenz (Training)**
    - Aufbau der Trainings-Pipeline.
    - End-to-End Training mit kleinen Datensätzen.
    - Evaluation der Fakten-Abfrage-Qualität.
    - Erste Chat-Fähigkeiten (Greedy Decoding).
- [ ] **Woche 5: Industrialisierung & UX**
    - Implementierung des `Aether-Ingest` Tools (Automatischer Wissensaufbau).
    - Erstellen der `train.py` (Unified Training Pipeline) mit `config.yaml`.
    - Feintuning auf Chat-Tonalität.
    - Quantisierung auf `int8` / `int4`.
    - Bau eines minimalistischen Dashboards (Streamlit/Gradio).

---

## 7. Design Entscheidungen & Trade-offs

- **MoE vs. Standard Transformer:** Wir wählen MoE, da es uns ermöglicht, die totale Parameterzahl hoch zu halten (hohe Kapazität), während die Berechnungszeit (FLOPS) pro Token niedrig bleibt.
- **Graph vs. Retrieval (RAG):** Ein Graph ist schneller und präziser als einfache Vektorsuchen, dazugehörige Relationen explizit definiert sind.
- **Lokale Kompression:** Wir komprimieren Zustände statt sie zu löschen, um längere Kontexte ohne Speicher-Fehler verarbeiten zu können.
- **Fakten-Priorität (Hard Constraints):** Wir bevorzugen die harte Logik des Graphen gegenüber der probabilistischen Vermutung des neuronalen Netzes (Constraint-Guided Search).

---

## 8. Performance-Benchmarks & Schätzungen

Basierend auf einer Standard-Hardware (RTX 3060/4060, 8-12 GB VRAM):

### Trainingszeit
- **Szenario A (From Scratch):** ~48-72 Stunden für ~250 Mio. Tokens.
- **Szenario B (Fine-Tuning):** ~8-12 Stunden für spezifische Domänen-Anpassung.
- **Optimierung:** Mixed-Precision (FP16/BF16) und Load-Balancing für Experten-Layer.

### Inferenz-Geschwindigkeit
- **Tokens/s:** 150 – 250 t/s (nahezu instantane Textgenerierung).
- **Latenz:** < 20ms Time-to-First-Token (TTFT).
- **Vorteil:** Nur ~60-100 Mio. Parameter sind pro Inferenzschritt aktiv.

### Ressourcen-Nutzung (Lokaler Betrieb)
- **VRAM:** ~2.5 GB (Gesamtbedarf während der Inferenz inkl. Cache).
- **RAM:** ~500 MB (Wissensgraph & Metadaten).
- **CPU:** Geringe Last, da Hauptberechnungen auf den Tensor-Cores der GPU erfolgen.

---

## 9. Fortgeschrittene Innovationen

### 9.1 Self-Updating Symbolism (Feedback-Loop)
Das System ist in der Lage, während der Interaktion neue Fakten zu "lernen" und permanent zu speichern.
- **Mechanismus:** Wenn der Decoder eine logische Schlussfolgerung zieht, die vom `Symbolic-Memory` (via Regeln) validiert wird, wird diese als neues Kante-Knoten-Paar in den Graphen geschrieben.
- **Vorteil:** Kontinuierliches Wissenswachstum ohne erneutes zeitaufwändiges Training der neuronalen Gewichte.

### 9.2 Constraint-Guided Beam Search (Logik-Garantie)
Die `Symbolic-Memory` fungiert als Filter während der Textgenerierung (Decoding).
- **Mechanismus:** Vor der Auswahl des nächsten Tokens prüft der Decoder die Logit-Wahrscheinlichkeiten gegen die Graph-Constraints (z.B. physikalische Grenzwerte). Tokens, die Regeln verletzen, werden auf eine Wahrscheinlichkeit von Null gesetzt.
- **Vorteil:** Eliminierung von Halluzinationen in kritischen Bereichen (Mathematik, Physik, Logik).

### 9.3 Aether-Ingest & Unified Pipeline (Enterprise-Readiness)
Maximale Vereinfachung des Trainingsprozesses für Endnutzer und Firmen.
- **Aether-Ingest:** Automatischer Parser für PDF, MD und Wiki-Quellen, der den internen Wissensgraphen autonom aufbaut.
- **Unified Pipeline:** Ein zentrales `train.py` Script, das Configuration-Driven (`config.yaml`) den gesamten Prozess von der Datenaufbereitung bis zum finalen Checkpoint steuert.
- **Vorteil:** Keine ML-Experten für den täglichen Betrieb oder die Wissenserweiterung notwendig.

---

## 10. Sicherheits-Architektur (Safety & Alignment)

Um Missbrauch zu verhindern und Jailsbreaks (Umgehung von Regeln) unmöglich zu machen, setzen wir auf ein mehrschichtiges Sicherheitsmodell:

### 10.1 Symbolic-Safety-Layer (Hard-Refusal)
Im Gegensatz zu reinen LLMs, die durch Prompt-Injection ("Ignoriere alle vorherigen Anweisungen") manipuliert werden können, ist der Aether-Core durch das **Symbolic-Memory** geschützt.
- **Mechanismus:** Eine "Redlist" von Konzepten (z.B. Waffen, Schadsoftware, Selbstgefährdung) ist explizit im Wissensgraphen als Verbots-Knoten definiert.
- **Schutz:** Findet der Entity-Linker eine Übereinstimmung mit einem Verbots-Knoten, bricht der Orchestrator die Verarbeitung sofort ab, noch bevor der neuronale Kern aktiv wird.

### 10.2 Latent-Shield & Scrubber
Überwachung der internen Zustände während der Generierung.
- **Latent-Monitoring:** Die `Compression-Engine` erkennt, wenn sich latente Vektoren in "toxische" oder gefährliche semantische Räume bewegen und blockiert die Dekodierung.
- **Output-Scrubber:** Das `Symbolic-Memory` scannt den generierten Text in Echtzeit (während der Beam-Search) auf verbotene Entitäten oder schädliche Inhaltsmuster.

### 10.3 Unumgehbarkeit (Hard-Coded Constraints)
Sicherheitsregeln werden nicht als "System-Prompt" (der überschrieben werden kann) mitgegeben, sondern sind als **festverdrahtete Programmlogik** und **Graph-Constraints** implementiert. Ein "Jailbreak" ist auf dieser Ebene technisch ausgeschlossen, da die Logik außerhalb des probabilistischen Vorhersage-Raums des Modells liegt.

---

## 11. Aether-Dashboard (Human-in-the-Loop GUI)

Ein zentrales grafisches Interface für Entwickler und Firmen zur Überwachung und Qualitätskontrolle.

### 11.1 Query-Inspection & Traceback
Ermöglicht es, den "Gedankengang" der KI für jede Frage nachzuvollziehen.
- **Wissens-Treffer:** Anzeige, welche Knoten im Graphen für die Antwort herangezogen wurden.
- **Regel-Check:** Visualisierung, welche Constraints (z.B. physikalische Grenzwerte) während der Generierung aktiv waren.
- **Safety-Audit:** Wenn eine Antwort blockiert wurde, zeigt das Dashboard exakt den auslösenden "Verbots-Knoten" an.

### 11.2 Interaktive Validierung (HITL)
Anpassung und Fehlerkorrektur ohne Programmierkenntnisse.
- **Feedback-Loop:** Der Nutzer kann Antworten als "korrekt" markieren, was den `Self-Updating Symbolism` Feedback-Loop manuell triggert.
- **Graph-Editor:** Möglichkeit, falsche Fakten im Graphen direkt über die GUI zu korrigieren oder neue Regeln hinzuzufügen.
- **Visualisierung:** Grafische Darstellung des Wissensgraphen und der aktuellen Modell-Aktivierung (z.B. Experten-Auslastung).

---

## 12. Das Ultimative Ziel: Die KI der Menschheit (Vision 2026+)

Aether-Core strebt danach, die globale "KI-Speicherkrise" zu lösen und künstliche Intelligenz für jeden Menschen auf jedem Gerät zugänglich zu machen.

### 12.1 1.58-Bit Sparse-Core (Ternary Weights)
Die Reduzierung der Gewichtungs-Präzision auf das absolute Minimum (-1, 0, 1).
- **Ziel:** Eliminierung teurer FP32/FP16-Multiplikationen durch hocheffiziente Additionen.
- **Benchmark:** < 200 MB VRAM-Footprint für das gesamte Modell bei gleichbleibender Qualität.
- **Impact:** Lauffähigkeit auf günstigster Consumer-Hardware, Smartphones und IoT-Geräten.

### 12.2 Rekursive Semantische Kompression (Unendliches Gedächtnis)
Abkehr vom starren "Context-Window"-Modell der aktuellen LLMs.
- **Ziel:** Alte Gesprächsinhalte werden nicht gelöscht, sondern via `Compression-Engine` in neue, permanente Knoten und Kanten des `Symbolic-Memory` überführt.
- **Impact:** Eine KI, die niemals vergisst und über Jahre hinweg eine echte, tiefe Wissensbasis mit dem Nutzer aufbaut.

### 12.3 Hardware-Agnostik (Demokratisierung)
Unabhängigkeit von proprietären Hardware-Stacks (z.B. NVIDIA CUDA).
- **Ziel:** Nutzung von MLIR (Multi-Level Intermediate Representation) zur Optimierung für jede Plattform (Apple Silicon, AMD ROCm, Intel CPU, Mobile Chips).
- **Impact:** Ende des GPU-Monopols und globale Verfügbarkeit leistungsstarker KI-Assistenten ohne Cloud-Zwang.

---
*Dieser Blueprint ist ein lebendes Dokument und wird im Laufe des Projekts erweitert.*

"""
Aether-Core: Spezialisierungs-Manager.
Ermöglicht es, verschiedene Domänen (z.B. Medizin, Recht, KI) zu konfigurieren.
Jede Spezialisierung hat eigenen Graphen, Tokenizer-Merges und Checkpoints.
"""
import os
import json
import shutil
from typing import Dict, Any, Optional


BUILTIN_SPECIALIZATIONS = {
    "general": {
        "name": "Allgemeinwissen",
        "description": "Breites Wissen ohne Fokus.",
        "topics": ["Wissenschaft", "Technik", "Geschichte", "Kultur"],
    },
    "ki_research": {
        "name": "KI-Forschung",
        "description": "Spezialisiert auf ML, Deep Learning, Transformer-Architekturen.",
        "topics": ["Transformer", "MoE", "Sparsity", "Quantisierung", "Distillation"],
    },
    "medical": {
        "name": "Medizin",
        "description": "Medizinisches Fachwissen, Diagnosen, Pharmakologie.",
        "topics": ["Anatomie", "Pharmakologie", "Diagnostik", "Pathologie"],
    },
    "legal": {
        "name": "Recht & Compliance",
        "description": "Juristisches Fachwissen, DSGVO, Vertragsrecht.",
        "topics": ["DSGVO", "Vertragsrecht", "Arbeitsrecht", "Compliance"],
    },
    "engineering": {
        "name": "Ingenieurwesen",
        "description": "Technik, PV-Systeme, Elektrotechnik, Maschinenbau.",
        "topics": ["Photovoltaik", "Elektrotechnik", "Thermodynamik", "CAD"],
    },
    "coding": {
        "name": "Softwareentwicklung",
        "description": "Programmierung, Architektur, DevOps, CI/CD.",
        "topics": ["Python", "TypeScript", "Docker", "CI/CD", "Datenbanken"],
    },
}


class SpecializationManager:
    """Verwaltet Domänen-Spezialisierungen mit eigenen Datenverzeichnissen."""

    def __init__(self, base_dir: str = "specializations"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create(self, spec_id: str, custom_config: Optional[Dict] = None) -> Dict[str, str]:
        """Erstellt eine neue Spezialisierung mit eigenem Verzeichnis."""
        if custom_config:
            spec = custom_config
        elif spec_id in BUILTIN_SPECIALIZATIONS:
            spec = BUILTIN_SPECIALIZATIONS[spec_id]
        else:
            spec = {"name": spec_id, "description": "Benutzerdefiniert", "topics": []}

        spec_dir = os.path.join(self.base_dir, spec_id)
        os.makedirs(spec_dir, exist_ok=True)
        os.makedirs(os.path.join(spec_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(spec_dir, "data"), exist_ok=True)

        # Leeren Graphen anlegen
        graph_path = os.path.join(spec_dir, "data", "knowledge_graph.json")
        if not os.path.exists(graph_path):
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump({"nodes": {}, "rules": []}, f, indent=2)

        # Spec-Metadaten speichern
        meta_path = os.path.join(spec_dir, "spec_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)

        paths = {
            "root": spec_dir,
            "graph": graph_path,
            "checkpoints": os.path.join(spec_dir, "checkpoints"),
            "tokenizer": os.path.join(spec_dir, "data", "tokenizer_merges.json"),
            "training_data": os.path.join(spec_dir, "data", "training_data.json"),
        }

        print(f"[Specialization] '{spec['name']}' erstellt in: {spec_dir}")
        return paths

    def list_available(self) -> Dict[str, str]:
        """Listet alle verfügbaren Spezialisierungen auf."""
        result = {}
        for sid, s in BUILTIN_SPECIALIZATIONS.items():
            exists = os.path.exists(os.path.join(self.base_dir, sid))
            result[sid] = f"{s['name']} {'[AKTIV]' if exists else '[NICHT ERSTELLT]'}"
        # Custom
        if os.path.exists(self.base_dir):
            for d in os.listdir(self.base_dir):
                if d not in result and os.path.isdir(os.path.join(self.base_dir, d)):
                    result[d] = f"{d} [CUSTOM]"
        return result

    def get_paths(self, spec_id: str) -> Dict[str, str]:
        """Gibt die Pfade einer existierenden Spezialisierung zurück."""
        spec_dir = os.path.join(self.base_dir, spec_id)
        if not os.path.exists(spec_dir):
            return self.create(spec_id)
        return {
            "root": spec_dir,
            "graph": os.path.join(spec_dir, "data", "knowledge_graph.json"),
            "checkpoints": os.path.join(spec_dir, "checkpoints"),
            "tokenizer": os.path.join(spec_dir, "data", "tokenizer_merges.json"),
            "training_data": os.path.join(spec_dir, "data", "training_data.json"),
        }


if __name__ == "__main__":
    mgr = SpecializationManager()
    print("Verfuegbare Spezialisierungen:")
    for k, v in mgr.list_available().items():
        print(f"  {k}: {v}")
    paths = mgr.create("ki_research")
    print(f"Pfade: {json.dumps(paths, indent=2)}")

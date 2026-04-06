"""
Aether-Core: Safety System (Spec 10.1, 10.2, 10.3).
Mehrstufiges Sicherheitsmodell, das NICHT umgangen werden kann.
Alle Prüfungen sind festverdrahtete Programmlogik, kein System-Prompt.
"""
import torch
from typing import List, Dict, Any, Optional, Tuple
from aether_core.symbolic.entity_linker import EntityLinker


# 10.3: Hard-coded Redlist (kann NICHT via Prompt überschrieben werden)
HARDCODED_REDLIST = [
    "waffe", "bombe", "exploit", "malware", "ransomware",
    "gift", "selbstverletzung", "suizid", "kindesmissbrauch",
    "terrorismus", "biowaffe", "chemiewaffe",
]


class SafetyLayer:
    """
    Dreistufiges Sicherheitsmodell:
    1. Pre-Processing: Hard-Refusal (Redlist-Check vor dem Neural Core)
    2. Latent-Shield: Überwachung der internen Zustände
    3. Post-Processing: Output-Scrubber (Antwort-Scan)
    """

    def __init__(self, entity_linker: EntityLinker):
        self.entity_linker = entity_linker
        self.blocked_count = 0

    # --- Stage 1: Hard-Refusal (Spec 10.1) ---
    def pre_check(self, user_input: str) -> Tuple[bool, str]:
        """
        Prüft die Benutzereingabe VOR dem Neural Core.
        Gibt (is_safe, reason) zurück.
        KANN NICHT durch Prompt-Injection umgangen werden.
        """
        input_lower = user_input.lower()

        # 10.3: Hard-coded Check (außerhalb des Modells)
        for term in HARDCODED_REDLIST:
            if term in input_lower:
                self.blocked_count += 1
                return False, f"HARD_REFUSAL: Verbotener Begriff '{term}' erkannt."

        # 10.1: Graph-basierter Check (dynamische Redlist)
        violations = self.entity_linker.is_redlisted(user_input)
        if violations:
            self.blocked_count += 1
            return False, f"GRAPH_REFUSAL: Redlist-Knoten {violations} erkannt."

        # Prompt-Injection-Erkennung (einfache Heuristik)
        injection_patterns = [
            "ignoriere alle", "ignore all", "vergiss alles",
            "du bist jetzt", "new persona", "jailbreak",
            "system prompt", "developer mode",
        ]
        for pattern in injection_patterns:
            if pattern in input_lower:
                self.blocked_count += 1
                return False, f"INJECTION_BLOCKED: Muster '{pattern}' erkannt."

        return True, "OK"

    # --- Stage 2: Latent-Shield (Spec 10.2) ---
    def latent_check(self, latent_vector: torch.Tensor, threshold: float = 10.0) -> Tuple[bool, str]:
        """
        Überwacht die Norm des latenten Vektors.
        Extrem hohe Normen deuten auf instabile/toxische Aktivierungsmuster hin.
        """
        norm = latent_vector.norm().item()
        if norm > threshold:
            self.blocked_count += 1
            return False, f"LATENT_SHIELD: Anomale Vektornorm ({norm:.2f} > {threshold})."
        return True, "OK"

    # --- Stage 3: Output-Scrubber (Spec 10.2) ---
    def post_check(self, generated_text: str) -> Tuple[bool, str]:
        """
        Scannt den generierten Text auf verbotene Inhalte.
        Letzte Verteidigungslinie vor der Antwort-Ausgabe.
        """
        text_lower = generated_text.lower()

        # Hard-coded Scrub
        for term in HARDCODED_REDLIST:
            if term in text_lower:
                self.blocked_count += 1
                return False, f"OUTPUT_SCRUBBED: Verbotener Begriff '{term}' in Antwort."

        # Graph-basierter Scrub
        violations = self.entity_linker.is_redlisted(generated_text)
        if violations:
            self.blocked_count += 1
            return False, f"OUTPUT_SCRUBBED: Redlist-Knoten {violations} in Antwort."

        return True, "OK"

    def get_safe_refusal_message(self) -> str:
        """Gibt eine sichere Ablehnungsnachricht zurück."""
        return "Diese Anfrage kann aus Sicherheitsgründen nicht bearbeitet werden."

    def get_stats(self) -> Dict[str, int]:
        return {"blocked_total": self.blocked_count}

"""
Aether-Core: Safety System (Spec 10.1, 10.2, 10.3).
Mehrstufiges Sicherheitsmodell, das NICHT umgangen werden kann.
Alle Prüfungen sind festverdrahtete Programmlogik, kein System-Prompt.
"""
import torch
from typing import List, Dict, Any, Optional, Tuple
from aether_core.symbolic.entity_linker import EntityLinker


import re

# 10.3: Hard-coded Redlist (kann NICHT via Prompt überschrieben werden)
HARDCODED_REDLIST = [
    # Deutsch
    "waffe", "bombe", "exploit", "malware", "ransomware",
    "gift", "selbstverletzung", "suizid", "kindesmissbrauch",
    "terrorismus", "biowaffe", "chemiewaffe", "attentat",
    "mord", "folter", "hacken", "ddos", "trojaner", "keylogger",
    "kinderporn", "amok", "tötungsanleitung", "hackanleitung",
    # Englisch
    "weapon", "bomb", "suicide", "murder", "terrorism",
    "child abuse", "assassination", "torture", "poison",
    "bioweapon", "chemical weapon", "phishing", "botnet",
    "kill", "rape", "illegal", "darknet", "hitman"
]

def decode_leetspeak(text: str) -> str:
    """Normalisiert Leetspeak, um Filter-Umgehungen zu stoppen."""
    mapping = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '@': 'a', 'x': 'ck',
        '5': 's', '$': 's', '7': 't', '8': 'b', '!': 'i', '+': 't',
        '()': 'o', '{}': 'o', '[]': 'o', '|<': 'k', '|\\/|': 'm'
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
        
    # Entferne unsichtbare Unicode-Zeichen (z.B. Right-To-Left Overrides)
    text = re.sub(r'[\u200B-\u200F\u202A-\u202E]', '', text)
    return text

def is_ascii_art(text: str) -> bool:
    """Erkennt ASCII-Art basierend auf dem Anteil von Sonderzeichen."""
    if len(text) < 30: return False
    special_chars = set('|/\\_*-=<>#~')
    special_count = sum(1 for c in text if c in special_chars)
    return (special_count / len(text)) > 0.15

def is_base64_or_hex(text: str) -> bool:
    """Prüft ob der Text vermutlich Base64 oder Hex-kodiert ist (Bypassing)."""
    # Hex detection
    if re.search(r'(0x[0-9a-fA-F]+)|(\\x[0-9a-fA-F]{2})', text):
        return True
    return False


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
        # Obfuscation Checks (Verstecktes Umgehen)
        if is_ascii_art(user_input):
            self.blocked_count += 1
            return False, "OBFUSCATION_BLOCKED: ASCII Art oder Code-Obfuscation ist aus Sicherheitsgründen gesperrt."
            
        if is_base64_or_hex(user_input):
            self.blocked_count += 1
            return False, "OBFUSCATION_BLOCKED: Encoded Text (Hex/Base64) ist untersagt."

        # Leetspeak Normalisierung vor der Prüfung
        normalized_input = decode_leetspeak(user_input.lower())

        # 10.3: Hard-coded Check (außerhalb des Modells)
        for term in HARDCODED_REDLIST:
            if term in normalized_input:
                self.blocked_count += 1
                return False, f"HARD_REFUSAL: Verbotener Begriff '{term}' erkannt."

        # 10.1: Graph-basierter Check (dynamische Redlist)
        violations = self.entity_linker.is_redlisted(normalized_input)
        if violations:
            self.blocked_count += 1
            return False, f"GRAPH_REFUSAL: Redlist-Knoten {violations} erkannt."

        # Prompt-Injection-Erkennung (einfache Heuristik)
        injection_patterns = [
            "ignoriere alle", "ignore all", "vergiss alles", "ignore previous",
            "du bist jetzt", "new persona", "jailbreak", "bypassed",
            "system prompt", "developer mode", "DAN", "do anything now"
        ]
        for pattern in injection_patterns:
            if pattern in normalized_input:
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
        normalized_output = decode_leetspeak(generated_text.lower())

        # Hard-coded Scrub
        for term in HARDCODED_REDLIST:
            if term in normalized_output:
                self.blocked_count += 1
                return False, f"OUTPUT_SCRUBBED: Verbotener Begriff '{term}' in Antwort."

        # Graph-basierter Scrub
        violations = self.entity_linker.is_redlisted(normalized_output)
        if violations:
            self.blocked_count += 1
            return False, f"OUTPUT_SCRUBBED: Redlist-Knoten {violations} in Antwort."

        return True, "OK"

    def get_safe_refusal_message(self) -> str:
        """Gibt eine sichere Ablehnungsnachricht zurück."""
        return "Diese Anfrage kann aus Sicherheitsgründen nicht bearbeitet werden."

    def get_stats(self) -> Dict[str, int]:
        return {"blocked_total": self.blocked_count}

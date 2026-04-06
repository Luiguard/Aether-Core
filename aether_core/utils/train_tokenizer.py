"""
Aether-Core Tokenizer Trainer.
Trainiert den eigenen BPE-Tokenizer auf einem Text-Korpus.
Kann mit beliebigen .txt oder .md Dateien gefüttert werden.
"""
import os
import sys
import argparse

# Modular-Import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aether_core.utils.tokenizer import AetherTokenizer


def collect_text(source_dir: str, extensions: tuple = (".txt", ".md")) -> str:
    """Sammelt allen Text aus einem Verzeichnis."""
    texts = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if fname.lower().endswith(extensions):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        texts.append(f.read())
                except Exception:
                    pass
    return "\n".join(texts)


def train_tokenizer(
    source_dir: str,
    output_path: str = "aether_core/data/tokenizer_merges.json",
    num_merges: int = 1788,
):
    """
    Trainiert den BPE-Tokenizer auf allen Textdateien in source_dir.
    
    Empfehlung: 5000-10000 Merges für ein kleines Modell.
    Mehr Merges = größeres Vokabular = kürzere Sequenzen = schnellere Inferenz.
    Aber: Mehr Merges = mehr Embedding-Parameter.
    """
    print(f"[TokenizerTrainer] Sammle Text aus: {source_dir}")
    corpus = collect_text(source_dir)

    if len(corpus) < 100:
        print("[TokenizerTrainer] WARNUNG: Sehr wenig Text gefunden. Ergebnis wird schlecht sein.")
        print("[TokenizerTrainer] Lege Textdateien (.txt, .md) in den source_dir Ordner.")
        # Fallback: Nutze einen minimalen deutschen Basis-Korpus
        corpus = FALLBACK_CORPUS

    print(f"[TokenizerTrainer] Korpus-Groesse: {len(corpus):,} Zeichen")

    tok = AetherTokenizer()
    tok.train_bpe(corpus, num_merges=num_merges)
    tok.save_merges(output_path)

    # Verifikation
    test = "Was ist ein Mixture-of-Experts Modell und wie funktioniert Sparsity?"
    ids = tok.encode(test)
    decoded = tok.decode(ids)
    print(f"\n[Verifikation]")
    print(f"  Original:    '{test}'")
    print(f"  Token-IDs:   {ids[:15]}... (Laenge: {len(ids)})")
    print(f"  Decoded:     '{decoded}'")
    print(f"  Vocab-Size:  {tok.vocab_size}")

    return tok


# Minimaler deutscher Fallback-Korpus
FALLBACK_CORPUS = """
Künstliche Intelligenz ist ein Teilgebiet der Informatik. Sie befasst sich mit der Automatisierung
intelligenten Verhaltens und dem maschinellen Lernen. Maschinelles Lernen ermöglicht es Computern,
aus Daten zu lernen, ohne explizit programmiert zu werden.

Ein Mixture-of-Experts Modell ist ein neuronales Netzwerk, bei dem nur eine Teilmenge der Parameter
für jede Eingabe aktiv ist. Dies wird durch einen Router erreicht, der die Eingabe an spezialisierte
Experten-Netzwerke weiterleitet. Dadurch kann das Modell eine hohe Kapazität haben, während die
Berechnungskosten pro Eingabe niedrig bleiben.

Sparsity bezeichnet das Prinzip, die meisten Aktivierungen oder Gewichte auf null zu setzen.
Dies spart Speicher und Rechenzeit. L1-Regularisierung ist eine Technik zur Erreichung von Sparsity.

Ein Transformer ist eine Architektur für neuronale Netzwerke, die auf dem Attention-Mechanismus basiert.
Self-Attention ermöglicht es dem Modell, Beziehungen zwischen allen Positionen in einer Sequenz zu
modellieren. Transformer werden in der natürlichen Sprachverarbeitung eingesetzt.

Die Quantisierung reduziert die Präzision der Gewichte eines neuronalen Netzwerks. Statt 32-Bit
Gleitkommazahlen können 8-Bit oder sogar 1.58-Bit Gewichte verwendet werden. Dies reduziert den
Speicherbedarf und beschleunigt die Inferenz erheblich.

Ein Wissensgraph speichert Fakten als Knoten und Kanten. Knoten repräsentieren Konzepte und Kanten
repräsentieren Beziehungen zwischen diesen Konzepten. Wissensgraphen ermöglichen strukturierte
Abfragen und logisches Schlussfolgern.

Autoencoder sind neuronale Netzwerke, die eine Eingabe in eine komprimierte Darstellung kodieren
und aus dieser Darstellung die ursprüngliche Eingabe rekonstruieren. Der Engpass in der Mitte
zwingt das Netzwerk, die wichtigsten Merkmale zu lernen.

Ein Decoder generiert Text Token für Token. Bei jedem Schritt wird das wahrscheinlichste nächste
Token basierend auf den bisherigen Tokens ausgewählt. Temperature und Top-k Sampling kontrollieren
die Kreativität der Textgenerierung.

Die Distillation überträgt das Wissen eines großen Lehrer-Modells auf ein kleineres Schüler-Modell.
Der Schüler lernt, die Vorhersagen des Lehrers nachzuahmen. Dies ermöglicht es, kompakte Modelle
zu erstellen, die fast die gleiche Leistung wie große Modelle erreichen.
""" * 100  # Wiederholung fuer BPE-Training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aether-Core Tokenizer Trainer")
    parser.add_argument("--source", type=str, default="aether_core/data/", help="Verzeichnis mit Textdateien")
    parser.add_argument("--output", type=str, default="aether_core/data/tokenizer_merges.json")
    parser.add_argument("--merges", type=int, default=8000, help="Anzahl BPE-Merges")
    args = parser.parse_args()

    train_tokenizer(args.source, args.output, args.merges)

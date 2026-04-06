import torch
import yaml
import os
import argparse
from aether_core.symbolic.symbolic_memory import SymbolicMemory
from aether_core.utils.tokenizer import AetherTokenizer
from aether_core.utils.ingest import AetherIngest
from aether_core.utils.distiller import ModelDistiller
from aether_core.utils.factory import build_models
from aether_core.utils.scaling import print_scaling_report


class AetherOrchestrator:
    """
    Aether-Core Orchestrator.
    Nutzt die Model-Factory für automatische Skalierung.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        print(f"--- Aether-Core Orchestrator v{self.config['version']} ---")

        # Hardware
        self.device = torch.device(self.config["device"] if torch.cuda.is_available() else "cpu")
        print(f"[Aether-Core] Geraet: {self.device}")

        # Tokenizer
        merges_path = "aether_core/data/tokenizer_merges.json"
        self.tokenizer = AetherTokenizer(merges_path if os.path.exists(merges_path) else None)

        # Symbolic-Memory
        self.sm = SymbolicMemory(self.config["symbolic"]["graph_path"])

        # Model-Factory: Baut alles basierend auf Preset + VRAM
        self.snc, self.cd, self.ce, self.estimates, self.strategy = build_models(
            self.config, self.tokenizer.vocab_size, self.device
        )

        # Tools
        self.ingest = AetherIngest(self.config["symbolic"]["graph_path"])
        self.distiller = ModelDistiller(self.snc)

        print(f"[Aether-Core] System bereit. Vocab: {self.tokenizer.vocab_size}")

    def train_step(self, question: str):
        """Ein Trainings-Schritt."""
        token_ids = self.tokenizer.encode(question)
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        entities = list(self.sm.graph.get("nodes", {}).keys())
        sym_context = self.sm.get_context_for_question(
            entities, embedding_dim=self.config["neural"]["d_model"]
        ).to(self.device)

        hidden_state = self.snc(tokens, sym_context)
        z_latent = self.ce.encode(hidden_state[:, -1, :])
        logits = self.cd(tokens, sym_context)

        print(f"[Train] Tokens: {len(token_ids)} | Logits: {logits.shape}")

    def infer(self, question: str) -> str:
        """Inferenz: Frage -> Antwort."""
        token_ids = self.tokenizer.encode(question)
        prompt = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        entities = [k for k in self.sm.graph.get("nodes", {}).keys() if k.lower() in question.lower()]
        d_model = self.config["neural"]["d_model"]
        context_emb = self.sm.get_context_for_question(entities, d_model).to(self.device) if entities else None

        generated_ids = self.cd.generate(
            prompt_ids=prompt,
            max_new_tokens=128,
            temperature=0.7,
            eos_token_id=self.tokenizer.eos_token_id,
            context_emb=context_emb,
        )
        return self.tokenizer.decode(generated_ids[len(token_ids):])

    def run(self):
        """Modus-Dispatcher."""
        mode = self.config["mode"]

        if mode == "train":
            print("[Aether-Core] Starte Training...")
            self.train_step("Was ist ein Mixture-of-Experts Modell?")

        elif mode == "infer":
            print("[Aether-Core] Starte Inferenz...")
            print(f"[Inferenz] {self.infer('Was ist Sparsity?')}")

        elif mode == "ingest":
            self.ingest.process_file("demo_dokument.txt")

        elif mode == "distill":
            self.distiller.extract_from_mit_model("GPT-2-124M")

        elif mode == "api":
            print("[Aether-Core] Starte OpenAI-kompatiblen API-Server...")
            import uvicorn
            from aether_core.utils.api import app
            uvicorn.run(app, host="127.0.0.1", port=8444)

        elif mode == "report":
            preset = self.config.get("scaling", {}).get("preset", "small")
            vram = self.config.get("vram_gb", 12.0)
            print_scaling_report(preset, self.tokenizer.vocab_size, vram)

        else:
            print(f"[Aether-Core] Unbekannter Modus: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aether-Core Unified CLI")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    orchestrator = AetherOrchestrator(args.config)
    orchestrator.run()

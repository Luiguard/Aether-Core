# 🌟 Aether-Core: Autonomous Neuro-Symbolic AI

**Aether-Core** is a next-generation, local AI architecture that combines the reasoning and infinite memory capabilities of **Symbolic Knowledge Graphs** with the fluid language generation of **Sparse Neural Networks (MoE)**. Designed specifically for low-resource hardware, Aether-Core runs entirely on CPU or small GPUs (via 1.58-Bit Ternary Quantization) and can autonomously expand its knowledge base using a DeepSeek-Teacher API without requiring costly backpropagation for every new fact.

![Aether-Core UI](https://img.shields.io/badge/Aether-Nano_Ready-66fcf1?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)

## 🚀 Key Features

*   **Neuro-Symbolic Architecture:** A lightweight Mixture-of-Experts (MoE) transformer generates human-like language, while a dynamic Symbolic Knowledge Graph serves as its source of truth and memory limitlessly avoiding hallucinations.
*   **Zero-Shot Knowledge Updating (API-Learning):** Introduce new facts, rules, and ontology via the integrated REST API or Chat. Aether learns via "Knowledge Injection" directly into the Graph - zero model re-training and zero GPU hours needed.
*   **Autonomous Learning Agent:** A completely background-running Observer process detects knowledge gaps during interactions or idle states, queries DeepSeek autonomously for structured facts, and populates the graph without human interaction.
*   **1.58-Bit Ternary Quantization:** Built-in VAE compression and {-1, 0, +1} weight optimization strictly cuts memory usage down allowing massive language models to sit effectively on consumer grade SSD/RAM/VRAM hybrids.
*   **Premium Web-UI with API Controller:** A stunning Glassmorphism Interface out-of-the-box allows simple user interactions, built-in DeepSeek API-Key management, und background distillation triggering.

## 🧠 How it Works

The architecture follows a strict 3-tier pipeline:
1.  **Symbolic Memory (Graph):** Stores facts, concepts, and multi-relational rules.
2.  **Teacher Integration (DeepSeek):** The background orchestrator validates data logic via a strict schema prompt and imports it directly from cloud models (Teacher -> Student pattern).
3.  **Neural Decoder:** Translates tensor-embeddings taken directly from Graph-Nodes back into highly fluent, causal linguistic sequences.

## 🛠️ Installation & Setup

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/YOURNAME/Aether-Core.git
   cd Aether-Core
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the Engine:**
   Double click the `run_aether.bat` script on Windows or execute the launcher directly:
   ```bash
   python aether_launcher.py
   ```
   > The Web-UI and Neural Server run concurrently and manage everything (even auto-learning loop) for you.

## 🎓 Knowledge Distillation

Aether comes with a pre-configured training distillation script. Instead of training via massive, disorganized datasets, Aether-Core queries complex conversational structures specifically tailored for your required ontology and refines its grammatical weights.
* Use the **Settings Icon** inside the Chat UI to enter the targeted `epochs`.
* Start distillation directly from the browser! 

## 🛡️ Privacy & Safety
Aether-Core incorporates a state-of-the-art deterministic 3-Phase Safety Layer:
1. **Pre-Check Lexer:** Hardcoded Entity Linker blocks Redlist tokens before they reach the graph.
2. **Latent Shield:** Protects the memory nodes during ingestion.
3. **Output Scrubber:** Cleanses the neural representation prior to generating responses.

## 📜 Roadmap & Future Visions
- [ ] Integration of Recursive Semantic Compression (Auto-Compressing Chat histories into structural long-term facts).
- [ ] VSCode / Cursor API extension layer for seamless coding assistance.
- [ ] Fully offline Desktop-App compile utilizing PyInstaller.

## License
MIT License - Created to push the bounds of autonomous local systems.

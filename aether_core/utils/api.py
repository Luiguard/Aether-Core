"""
Aether-Core: OpenAI-kompatibler API-Server.
Implementiert /v1/chat/completions und /v1/models.
Kann direkt in VSCode (Continue, Cody, etc.) als Custom-Backend eingebunden werden.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import yaml
import json
import time
import asyncio

from aether_core.symbolic.symbolic_memory import SymbolicMemory
from aether_core.neural.moe import SparseCore
from aether_core.compression.engine import CompressionEngine
from aether_core.neural.decoder import ChatDecoder
from aether_core.utils.tokenizer import AetherTokenizer

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Aether-Core OpenAI-Compatible API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globale Initialisierung ---
config_path = "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Module laden
sm = SymbolicMemory(config["symbolic"]["graph_path"])
tokenizer = AetherTokenizer()  # Custom BPE

n_cfg = config["neural"]
c_cfg = config["compression"]

# Decoder (das Sprachmodul)
decoder = ChatDecoder(
    vocab_size=tokenizer.vocab_size,
    d_model=n_cfg["d_model"],
    n_layers=4,
    n_heads=n_cfg.get("n_heads", 12),
).to(device)

print(f"[Aether-API] Server bereit auf {device}. Vocab: {tokenizer.vocab_size}")


# --- Pydantic Models (OpenAI-kompatibel) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "aether-core"
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# --- API-Learning Endpunkte (behalten aus vorheriger Phase) ---
class NodeIn(BaseModel):
    id: str
    name: str
    properties: Dict[str, Any] = {}

class EdgeIn(BaseModel):
    source_id: str
    target_id: str
    relation_type: str

class FactIn(BaseModel):
    id: str
    key: str
    value: Any

class RuleIn(BaseModel):
    id: str
    type: str
    details: Dict[str, Any]

@app.post("/node")
async def create_node(node: NodeIn):
    if sm.add_node(node.id, node.name, node.properties):
        return {"status": "success", "message": f"Knoten {node.id} angelegt."}
    raise HTTPException(status_code=400, detail="Konflikt: Knoten existiert bereits.")

@app.post("/edge")
async def create_edge(edge: EdgeIn):
    if sm.add_edge(edge.source_id, edge.target_id, edge.relation_type):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Knoten nicht gefunden.")

@app.post("/fact")
async def create_fact(fact: FactIn):
    if sm.add_fact(fact.id, fact.key, fact.value):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Knoten nicht gefunden.")

@app.post("/rule")
async def create_rule(rule: RuleIn):
    if sm.add_rule(rule.id, rule.type, rule.details):
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Regel existiert bereits.")

@app.get("/graph")
async def get_graph():
    return sm.graph

class APIKeyIn(BaseModel):
    api_key: str

@app.post("/v1/settings/apikey")
async def set_api_key(data: APIKeyIn):
    import os
    os.environ["AETHER_TEACHER_API_KEY"] = data.api_key
    return {"status": "success", "message": "API Key gespeichert."}

class TrainRequest(BaseModel):
    epochs: int

@app.post("/v1/train/distill")
async def trigger_distill(req: TrainRequest, background_tasks: BackgroundTasks):
    # distill import hier um cycle dependencies zu vermeiden
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from distill import distill
    
    # Führe distill als asyncio / background task aus
    background_tasks.add_task(distill, "config.yaml", req.epochs)
    return {"status": "success", "message": f"Training gestartet. Log siehe Konsole."}

# --- OpenAI-kompatible Endpunkte ---
@app.get("/v1/models")
async def list_models():
    """Gibt verfügbare Modelle zurück (VSCode/Continue fragt das ab)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "aether-core",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "aether-core-local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-kompatibles Chat-Completion Interface.
    Nimmt Messages entgegen, generiert autoregressive Antwort.
    """
    # 1. Nachrichten zu einem Prompt zusammenbauen
    prompt_parts = []
    for msg in request.messages:
        prefix = {"system": "System", "user": "User", "assistant": "Assistant"}.get(msg.role, msg.role)
        prompt_parts.append(f"{prefix}: {msg.content}")
    prompt_parts.append("Assistant:")
    prompt_text = "\n".join(prompt_parts)

    # 2. Tokenisieren
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 3. Symbolischen Kontext extrahieren (einfache Keyword-Suche)
    all_text = " ".join(m.content for m in request.messages)
    entities = [k for k in sm.graph.get("nodes", {}).keys() if k.lower() in all_text.lower()]
    
    # --- ON-DEMAND LEARNING TRIGGER ---
    if not entities and len(request.messages) > 0:
        last_msg = request.messages[-1].content
        # Wenn wir gar kein Konzept gematcht haben, beauftragen wir den Teacher
        # Wir rufen den DeepSeekIntegrator auf, um das Konzept strukturiert in den Graph zu pumpen
        from aether_core.utils.teacher import TeacherClient
        from aether_core.utils.integrator import DeepSeekIntegrator
        
        print(f"[API] Lücke entdeckt für: '{last_msg}'. Initiiere On-Demand Teacher Learning...")
        try:
            # Wir extrahieren on-the-fly (synchron, da das UI lädt)
            teacher = TeacherClient()
            integrator = DeepSeekIntegrator(teacher, "http://127.0.0.1:8444")
            if integrator.acquire_topic(f"Details und Fakten zu: {last_msg}"):
                print("[API] Erfolgreich gelernt. Lade frischen Node...")
                # Lade Graph Update manuell einmalig nach dem POST (der Integrator nutzt REST POST)
                entities = [k for k in sm.graph.get("nodes", {}).keys() if k.lower() in all_text.lower()]
        except Exception as e:
            print(f"[API] Fehler beim On-Demand Learning: {e}")
    # ----------------------------------
    
    context_emb = sm.get_context_for_question(entities, embedding_dim=n_cfg["d_model"]).to(device) if entities else None

    # 4. Generieren
    generated_ids = decoder.generate(
        prompt_ids=prompt_tensor,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        eos_token_id=tokenizer.eos_token_id,
        context_emb=context_emb,
    )

    # 5. Nur die neuen Tokens decodieren
    new_ids = generated_ids[len(prompt_ids):]
    response_text = tokenizer.decode(new_ids)

    # 6. OpenAI-kompatible Antwort bauen
    return ChatCompletionResponse(
        id=f"aether-{int(time.time())}",
        created=int(time.time()),
        model="aether-core",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(new_ids),
            total_tokens=len(prompt_ids) + len(new_ids),
        ),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device), "vocab_size": tokenizer.vocab_size}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8444)

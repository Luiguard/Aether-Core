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
import os
import sys

from aether_core.symbolic.symbolic_memory import SymbolicMemory
from aether_core.neural.moe import SparseCore
from aether_core.compression.engine import CompressionEngine
from aether_core.neural.decoder import ChatDecoder
from aether_core.utils.tokenizer import AetherTokenizer
from aether_core.symbolic.entity_linker import EntityLinker
from aether_core.symbolic.safety import SafetyLayer

from fastapi.middleware.cors import CORSMiddleware

from aether_core.utils.checkpoint import CheckpointManager

app = FastAPI(title="Aether-Core OpenAI-Compatible API", version="1.0.0")

LAST_INTERACTION_TIME = time.time()
IS_TRAINING = False
CKPT_MGR = CheckpointManager("checkpoints")

def load_latest_weights():
    """Lädt die neuesten Gewichte der Neural-Engine."""
    global decoder, snc, ce
    latest = CKPT_MGR.find_latest()
    if latest:
        print(f"[Aether-API] Lade Checkpoint: {latest}")
        CKPT_MGR.load(latest, snc, decoder, ce)
    else:
        print("[Aether-API] Keine Checkpoints gefunden. Starte mit initialen Gewichten.")

def run_distill_safe(epochs=20):
    global IS_TRAINING, LAST_INTERACTION_TIME
    if IS_TRAINING: return
    IS_TRAINING = True
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from distill import distill
        distill("config.yaml", epochs)
        # Nachdem Training fertig ist, Gewichte in-place reloaden
        load_latest_weights()
    except Exception as e:
        print(f"[Training] Fehler: {e}")
    finally:
        IS_TRAINING = False
        LAST_INTERACTION_TIME = time.time()

@app.on_event("startup")
async def start_idle_observer():
    asyncio.create_task(idle_loop())

async def idle_loop():
    from fastapi.concurrency import run_in_threadpool
    global LAST_INTERACTION_TIME, IS_TRAINING
    while True:
        await asyncio.sleep(10)
        if not IS_TRAINING and (time.time() - LAST_INTERACTION_TIME > 120):
            print("\n[IdleObserver] 2 Minuten Leerlauf erkannt! Auto-Distill startet (60 Epochen)...")
            await run_in_threadpool(run_distill_safe, 60)

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

n_cfg = config["neural"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Module laden
sm = SymbolicMemory(config["symbolic"]["graph_path"])
entity_linker = EntityLinker(sm.graph)
safety_layer = SafetyLayer(entity_linker)
# Tokenizer laden (muss mit distill.py konsistent sein)
merges_path = "aether_core/data/tokenizer_merges.json"
tokenizer = AetherTokenizer(merges_path if os.path.exists(merges_path) else None)

# Falls Vokabular noch klein ist (keine Merges), Basis-Spec einhalten
n_cfg = config["neural"]
c_cfg = config["compression"]

# Die Neural-Engine besteht aus SNC, Decoder und CE
snc = SparseCore(
    vocab_size=tokenizer.vocab_size,
    d_model=n_cfg["d_model"],
    n_layers=n_cfg["n_layers"],
    n_experts=n_cfg["moe"]["n_experts"],
    top_k=n_cfg["moe"]["top_k"],
).to(device)

decoder = ChatDecoder(
    vocab_size=tokenizer.vocab_size,
    d_model=n_cfg["d_model"],
    n_layers=4,
    n_heads=n_cfg.get("n_heads", 12),
).to(device)

ce = CompressionEngine(
    input_dim=n_cfg["d_model"],
    latent_dim=c_cfg["latent_dim"],
).to(device)

# Gewichte laden
load_latest_weights()

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
    tokens_per_second: float = 0.0
    elapsed_ms: float = 0.0

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
    """Speichert den API-Key permanent in der config.yaml."""
    import yaml
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        
        # Sektion sicherstellen
        if "teacher" not in cfg:
            cfg["teacher"] = {}
            
        cfg["teacher"]["api_key"] = data.api_key
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
            
        # Globalen Zustand aktualisieren
        os.environ["AETHER_TEACHER_API_KEY"] = data.api_key
        print(f"[Aether-API] Teacher API-Key permanent in config.yaml gespeichert.")
        return {"status": "success", "message": "API Key permanent gespeichert."}
    except Exception as e:
        return {"status": "error", "message": f"Konnte Key nicht speichern: {e}"}

class TrainRequest(BaseModel):
    epochs: int

@app.post("/v1/train/distill")
async def trigger_distill(req: TrainRequest, background_tasks: BackgroundTasks):
    if IS_TRAINING:
        return {"status": "error", "message": "Training läuft bereits!"}
    
    background_tasks.add_task(run_distill_safe, req.epochs)
    return {"status": "success", "message": f"Training (Epochen: {req.epochs}) gestartet. Log siehe Konsole."}

@app.get("/v1/system/status")
async def get_status():
    return {"is_training": IS_TRAINING}

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


def background_learn(topic: str):
    from aether_core.utils.teacher import TeacherClient
    from aether_core.utils.integrator import DeepSeekIntegrator
    print(f"\n[API Background] Starte autonome Akquise für: {topic}")
    try:
        teacher = TeacherClient()
        integrator = DeepSeekIntegrator(teacher, "http://127.0.0.1:8444")
        integrator.acquire_topic(f"Details und Fakten zu: {topic}")
    except Exception as e:
        print(f"[API Background] Fehler: {e}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    global LAST_INTERACTION_TIME
    LAST_INTERACTION_TIME = time.time()
    """
    OpenAI-kompatibles Chat-Completion Interface.
    Nimmt Messages entgegen, generiert autoregressive Antwort.
    """
    # 1. Check Safety Rules (Pre-Processing Spec 10.1)
    user_input_pure = " ".join(m.content for m in request.messages if m.role == "user")
    is_safe, refusal_reason = safety_layer.pre_check(user_input_pure)
    
    if not is_safe:
        return ChatCompletionResponse(
            id=f"aether-{int(time.time())}",
            created=int(time.time()),
            model="aether-core",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=f"⚠️ {refusal_reason}"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    # 1b. Nachrichten zu einem Prompt zusammenbauen
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
        print(f"[API] Lücke entdeckt für: '{last_msg}'. Beauftrage Background-Agent...")
        
        # Feuere es asynchron ab und antworte dem Nutzer sofort.
        background_tasks.add_task(background_learn, last_msg)
        
        response_text = f"Bisher fehlt mir zu '{last_msg}' leider das spezifische Wissen. Ich habe aber soeben erfolgreich einen autonomen Hintergrund-Agenten beauftragt, der sich dieses Thema über DeepSeek strukturiert aneignet. Bitte stelle mir die Frage in etwa einer Minute noch einmal!"
        
        return ChatCompletionResponse(
            id=f"aether-{int(time.time())}",
            created=int(time.time()),
            model="aether-core",
            choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=response_text), finish_reason="stop")],
            usage=ChatCompletionUsage(prompt_tokens=1, completion_tokens=30, total_tokens=31),
        )
    # ----------------------------------
    
    context_emb = sm.get_context_for_question(entities, embedding_dim=n_cfg["d_model"]).to(device) if entities else None

    # 4. Generieren
    start_gen = time.time()
    generated_ids = decoder.generate(
        prompt_ids=prompt_tensor,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        eos_token_id=tokenizer.eos_token_id,
        context_emb=context_emb,
    )
    end_gen = time.time()
    elapsed = max(end_gen - start_gen, 0.001)

    # 5. Nur die neuen Tokens decodieren
    new_ids = generated_ids[len(prompt_ids):]
    completion_tokens = len(new_ids)
    response_text = tokenizer.decode(new_ids).strip()
    
    tokens_per_second = completion_tokens / elapsed

    # --- EXPERIMENTAL QUALITY CHECK (Weiteres Training nötig?) ---
    needs_training = False
    words = response_text.split()
    
    if len(words) > 4:
        # Check 1: Zu viele Wiederholungen (Looping)
        max_repeats, current_repeats = 1, 1
        for i in range(1, len(words)):
            if words[i].lower() == words[i-1].lower():
                current_repeats += 1
                max_repeats = max(max_repeats, current_repeats)
            else:
                current_repeats = 1
        if max_repeats >= 4:
            needs_training = True

    # Check 2: Extrem lange Gibberish-Wörter ("Asernernernernern...")
    if any(len(w) > 25 for w in words):
        needs_training = True
        
    # Check 3: Leere Ausgabe (Model Collapse)
    if not response_text:
        needs_training = True

    if needs_training:
        if IS_TRAINING:
            response_text += "\n\n⚙️ *(System-Update: Die Neural-Engine optimiert sich gerade autonom. Bitte hab einen Moment Geduld, die Sprachqualität verbessert sich in Kürze.)*"
        else:
            response_text += "\n\n⚠️ *(System-Hinweis: Die Neural-Engine zeigt Anzeichen von Untrainiertheit (Degeneration). Bitte im Einstellungen-Zahnrad den 'Trainer starten', um die Sprachstruktur weiter zu destillieren.)*"

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
            completion_tokens=completion_tokens,
            total_tokens=len(prompt_ids) + completion_tokens,
            tokens_per_second=round(tokens_per_second, 2),
            elapsed_ms=round(elapsed * 1000, 2),
        ),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device), "vocab_size": tokenizer.vocab_size}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8444)

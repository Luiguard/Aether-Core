import os
import sys
import threading
import webbrowser
import time
import yaml
import uvicorn
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Setze Config explizit auf eine schlanke "Nano" Konfiguration für Endnutzer
def prepare_light_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # Modus auf API zwingen und kleinstes, schnellstes Modell wählen
        cfg["mode"] = "api"
        if "scaling" not in cfg:
            cfg["scaling"] = {}
        cfg["scaling"]["preset"] = "nano" # Nutzt nur ~100MB RAM, läuft überall
        cfg["device"] = "cpu" # Breitenmasse hat evtl keine dicke NVIDIA GPU
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
            
def start_api_server():
    from aether_core.utils.api import app
    # Läuft auf 8444
    uvicorn.run(app, host="127.0.0.1", port=8444, log_level="error")

class UIHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Cache verbieten für einfacheres Entwickeln/Reloaden
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        SimpleHTTPRequestHandler.end_headers(self)

def start_web_server():
    import functools
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    os.makedirs(web_dir, exist_ok=True)
    server_address = ('127.0.0.1', 8888)
    handler = functools.partial(UIHandler, directory=web_dir)
    httpd = HTTPServer(server_address, handler)
    httpd.serve_forever()

def start_continuous_distiller():
    """Überwacht den Datensatz und startet Training bei neuem Wissen."""
    print("[Distiller] Unendlicher Lernmodus aktiv (Auto-Distill).")
    data_path = "aether_core/data/training_data.json"
    last_size = 0
    if os.path.exists(data_path):
        last_size = os.path.getsize(data_path)
    
    while True:
        try:
            if os.path.exists(data_path):
                current_size = os.path.getsize(data_path)
                if current_size > last_size:
                    print(f"\n[Distiller] Neues Wissen erkannt ({current_size} Bytes). Starte Distillation...")
                    subprocess.run(["python", "distill.py", "20"], check=False)
                    last_size = current_size
            time.sleep(30) # Alle 30 Sekunden prüfen
        except Exception as e:
            print(f"[Distiller] Fehler: {e}")
            time.sleep(10)

def start_auto_agent():
    try:
        from aether_core.utils.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(api_url="http://127.0.0.1:8444")
        agent.run_loop(duration_s=86400, interval_s=60) # Läuft für 24 Stunden, jede Minute ein Request
    except Exception as e:
        print("[AutoAgent] Konnte nicht starten:", e)

if __name__ == "__main__":
    print("==================================================")
    print("🌟 AETHER-CORE LIGHT LAUNCHER 🌟")
    print("Für schwache Hardware und direkte Endnutzung.")
    print("==================================================")
    
    print("[1/5] Bereite Konfiguration vor (Modus: Nano / CPU)...")
    prepare_light_config()
    
    print("[2/5] Starte Aether Neural-Engine Server im Hintergrund...")
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    print("[3/5] Starte lokales Web-Interface...")
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    print("[4/5] Starte autonomen DeepSeek-Observer (Lern-Loop)...")
    # Wartet kurz, bis die API bereit ist
    time.sleep(4)
    agent_thread = threading.Thread(target=start_auto_agent, daemon=True)
    agent_thread.start()
    
    # 4b. Continuous Distiller
    distiller_thread = threading.Thread(target=start_continuous_distiller, daemon=True)
    distiller_thread.start()
    
    print("[5/5] Öffne Browser...")
    time.sleep(2) # Kurz warten, damit alles bootet
    webbrowser.open("http://127.0.0.1:8888/chat_ui.html")
    
    print("Fertig! Das System läuft. (Beenden mit STRG+C)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nAether-Core wird beendet. Bis bald!")
        sys.exit(0)

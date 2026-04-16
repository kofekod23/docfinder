# colab_helpers_cell.py
"""Single-cell Colab entrypoint. Assumes repo was git-cloned to /content/docfinder.

Ce script lance DEUX choses en parallèle sur Colab :
  1. Un serveur FastAPI `/encode` (port 8001) que le Mac appelle pour encoder
     les requêtes BGE-M3 à distance (Mac stateless — voir spec v2).
  2. Le pipeline d'indexation `run_pipeline()` qui traite le corpus du Mac.

Les deux partagent la même instance `BGEM3Wrapper` pour ne charger le modèle
qu'une seule fois dans la mémoire GPU T4 (fp16 ≈ 2.3 GB).

Variables d'environnement requises :
  - MAC_BASE_URL      : URL du tunnel Cloudflare qui pointe vers le Mac.
  - DOCFINDER_ROOT    : racine à indexer côté Mac (ex. /Users/julien/Documents).
  - COLAB_QUERY_TOKEN : secret partagé avec le Mac (X-Auth-Token).

Exposer le port 8001 via un 2ᵉ tunnel Cloudflare nommé (dashboard
Zero Trust → Networks → Tunnels), puis renseigner `COLAB_ENCODE_URL` et
`COLAB_QUERY_TOKEN` dans le `.env` du Mac.
"""
import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, "/content/docfinder")

import uvicorn

from colab import query_server
from colab.client import MacClient
from colab.embedder_v2 import BGEM3Wrapper
from colab.extractor import extract_text
from colab.pipeline import run_pipeline

MAC_BASE = os.environ["MAC_BASE_URL"]
ROOT = os.environ["DOCFINDER_ROOT"]
if not os.environ.get("COLAB_QUERY_TOKEN", "").strip():
    raise RuntimeError(
        "COLAB_QUERY_TOKEN must be set before launching the cell "
        "(same value as on the Mac .env)."
    )

mac = MacClient(MAC_BASE)
embedder = BGEM3Wrapper()
embedder._model_or_build()  # force load once here, avoids race with uvicorn startup
tokenizer_decode = lambda ids: embedder._model_or_build().tokenizer.decode(ids)

# Partage du wrapper avec le serveur d'encodage → un seul modèle en VRAM.
query_server.set_wrapper(embedder)

config = uvicorn.Config(
    query_server.app, host="0.0.0.0", port=8001, log_level="info", access_log=False,
)
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True, name="query-server")
thread.start()
print("[colab_helpers_cell] query_server lancé sur :8001 (thread daemon).")

tmp = Path(tempfile.mkdtemp())
ck = Path("/content/checkpoint_v2.json")


def extractor(path, mode):
    return extract_text(path, mode=mode)


# Jupyter/Colab ont déjà une event loop active → asyncio.run() refuse
# (RuntimeError "cannot be called from a running event loop" sur Python 3.12+).
# On crée une loop dédiée et on y exécute le pipeline. Elle vit en parallèle
# de celle de Jupyter sans conflit (loops isolées par thread).
_loop = asyncio.new_event_loop()
try:
    _loop.run_until_complete(run_pipeline(
        MAC_BASE, ROOT,
        mac_client=mac, extractor=extractor, embedder=embedder,
        tokenizer_decode=tokenizer_decode,
        tmp_dir=tmp, checkpoint_path=ck,
    ))
finally:
    _loop.close()

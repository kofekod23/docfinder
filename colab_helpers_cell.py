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
# BGE-M3 XLM-R token counter pour le chunker (spec §6) : évite le mismatch
# mots/tokens (1.69 tokens/mot en français) qui faisait déborder hard_max.
_xlmr_tokenizer = embedder._model_or_build().tokenizer
tokenize_len_bgem3 = lambda t: len(_xlmr_tokenizer.encode(t, add_special_tokens=False))

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


# Jupyter/Colab ont une event loop active dans le thread principal → même
# asyncio.new_event_loop().run_until_complete() refuse ("Cannot run the event
# loop while another loop is running", check global per-thread via
# _get_running_loop()). Seule porte de sortie : un thread dédié qui n'a pas
# de loop courante.
def _run_pipeline_threaded():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_pipeline(
            MAC_BASE, ROOT,
            mac_client=mac, extractor=extractor, embedder=embedder,
            tokenizer_decode=tokenizer_decode,
            tmp_dir=tmp, checkpoint_path=ck,
            tokenize_len=tokenize_len_bgem3,
        ))
    finally:
        loop.close()


_pipeline_thread = threading.Thread(
    target=_run_pipeline_threaded, name="pipeline", daemon=False,
)
_pipeline_thread.start()
print("[colab_helpers_cell] pipeline lancé (thread dédié). Join en cours…")
_pipeline_thread.join()
print("[colab_helpers_cell] pipeline terminé.")

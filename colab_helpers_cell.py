# colab_helpers_cell.py
"""Single-cell Colab entrypoint. Assumes repo was git-cloned to /content/docfinder."""
import asyncio, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, "/content/docfinder")

from colab.client import MacClient
from colab.embedder_v2 import BGEM3Wrapper
from colab.extractor import extract_text, decide_mode
from colab.pipeline import run_pipeline

MAC_BASE = os.environ["MAC_BASE_URL"]   # set to the Cloudflare URL
ROOT = os.environ["DOCFINDER_ROOT"]     # e.g. /Users/julien/Documents

mac = MacClient(MAC_BASE)
embedder = BGEM3Wrapper()
tokenizer_decode = lambda ids: embedder._model_or_build().tokenizer.decode(ids)

tmp = Path(tempfile.mkdtemp())
ck = Path("/content/checkpoint_v2.json")

def extractor(path, mode):
    return extract_text(path, mode=mode)

asyncio.run(run_pipeline(
    MAC_BASE, ROOT,
    mac_client=mac, extractor=extractor, embedder=embedder,
    tokenizer_decode=tokenizer_decode,
    tmp_dir=tmp, checkpoint_path=ck,
))

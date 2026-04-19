"""Watchdog cloudflared pour Colab — keeps tunnel #2 alive indefinitely.

Usage (dans une cellule Colab séparée, APRÈS cell 3 qui a stocké TUNNEL2_TOKEN) :

    import subprocess, threading, time, os

    TUNNEL2_TOKEN = ...  # déjà défini par cell 3

    # Kill l'ancien cloudflared (cell 3) pour repartir propre
    subprocess.run(['pkill', '-f', 'cloudflared.*tunnel.*run'], check=False)
    time.sleep(1)

    def _tunnel_watchdog():
        while True:
            log = open('/content/cloudflared.log', 'ab')
            proc = subprocess.Popen(
                ['cloudflared', 'tunnel', '--no-autoupdate', 'run', '--token', TUNNEL2_TOKEN],
                stdout=log, stderr=log,
            )
            print(f'[tunnel-watchdog] cloudflared started PID={proc.pid}', flush=True)
            rc = proc.wait()  # bloque jusqu'à crash
            print(f'[tunnel-watchdog] cloudflared died rc={rc}, restarting dans 5s...', flush=True)
            time.sleep(5)

    _wd = threading.Thread(target=_tunnel_watchdog, daemon=True, name='tunnel-watchdog')
    _wd.start()
    print('[tunnel-watchdog] démarré — cloudflared auto-restart à chaque crash')

Ou, encore plus simple, lance directement ce fichier avec TUNNEL2_TOKEN en env :

    !TUNNEL2_TOKEN=<token> python /content/docfinder/scripts/colab_tunnel_watchdog.py &

Note : le fichier lu en tant que script démarre le watchdog en foreground (bloquant).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def _spawn_cloudflared(token: str) -> subprocess.Popen:
    log = open("/content/cloudflared.log", "ab")
    return subprocess.Popen(
        ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", token],
        stdout=log,
        stderr=log,
    )


def main() -> None:
    token = os.environ.get("TUNNEL2_TOKEN", "").strip()
    if not token:
        sys.exit("TUNNEL2_TOKEN env var required")

    # Kill any lingering cloudflared before starting fresh
    subprocess.run(["pkill", "-f", "cloudflared.*tunnel.*run"], check=False)
    time.sleep(1)

    while True:
        proc = _spawn_cloudflared(token)
        print(
            f"[tunnel-watchdog] cloudflared started PID={proc.pid}",
            flush=True,
        )
        rc = proc.wait()
        print(
            f"[tunnel-watchdog] cloudflared died rc={rc}, restarting in 5s...",
            flush=True,
        )
        time.sleep(5)


if __name__ == "__main__":
    main()

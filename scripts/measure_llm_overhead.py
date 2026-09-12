"""Separate the model's latency from the HTTP client's, because most of the
published figure was the client's.

Every LLM latency in the first version of this write-up was measured around
`urllib.request.urlopen` against `http://localhost:11434`, and reported as the
cost of running a 7B model. On this machine that number was wrong by an order of
magnitude, for a reason that has nothing to do with inference:

    getaddrinfo("localhost") -> ::1, 127.0.0.1

`localhost` resolves to the IPv6 loopback first. Ollama binds IPv4 only, so
every call opens a connection to `::1`, waits for Windows to refuse it -- about
two seconds -- and only then falls back to `127.0.0.1`. Python's urllib does not
do happy-eyeballs, so the wait is paid in full on every request, serially.

The give-away was in the repository's own results table. The DSPy arm, which
sends *longer* prompts through LiteLLM's HTTP client, was reported at 412 ms
against 2,362 ms for the retrieval arm. A longer prompt cannot be six times
faster. The write-up explained that gap with a story about compile-time
demonstrations; the actual cause was that the two arms used different HTTP
clients, and only one of them was paying the IPv6 timeout.

This script measures the three quantities separately so the claim in the README
is about the model:

    1. a no-work request (`/api/tags`) over each hostname -- pure client cost;
    2. a real classification call over each hostname -- client plus model;
    3. the difference, which is what the model actually costs.

Usage:
    python scripts/measure_llm_overhead.py
    python scripts/measure_llm_overhead.py --n 30

Writes reports/metrics_llm_overhead.json.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from licitaciones.utils import load_config, save_json, setup_logging  # noqa: E402

HOSTS = ("localhost", "127.0.0.1")

# A real prompt of the shape the classifier sends, so the model does the same
# amount of work in both arms. Kept literal rather than imported, because this
# script must not depend on the warehouse existing.
PROMPT = (
    "Eres un clasificador de compras publicas chilenas. Debes asignar el "
    "segmento UNSPSC de dos digitos que corresponde a lo que se esta "
    "comprando.\n\n"
    "Segmentos posibles:\n10: plantas y animales\n15: combustibles\n"
    "24: manejo de materiales\n39: equipos electricos\n42: equipos medicos\n"
    "43: equipos informaticos\n47: equipos de limpieza\n72: servicios de "
    "construccion\n80: servicios de gestion\n86: servicios educativos\n\n"
    "Titulo de la licitacion:\nADQ. SERV. REPARACION VEH-. FISCAL BT-278\n\n"
    'Responde SOLO con JSON: {"segmento": "NN"}'
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=15, help="calls per arm")
    return p.parse_args()


def resolution(host: str, port: int) -> dict:
    """What the hostname resolves to, and what it costs to connect to each."""
    started = time.perf_counter()
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    lookup_ms = (time.perf_counter() - started) * 1000

    attempts = []
    for family, _, _, _, address in infos:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(10)
        started = time.perf_counter()
        try:
            sock.connect(address)
            attempts.append({"address": address[0], "ms": round(
                (time.perf_counter() - started) * 1000, 1), "connected": True})
        except OSError as exc:
            attempts.append({"address": address[0], "ms": round(
                (time.perf_counter() - started) * 1000, 1), "connected": False,
                "error": type(exc).__name__})
        finally:
            sock.close()
    return {
        "host": host,
        "getaddrinfo_ms": round(lookup_ms, 1),
        "order": [i[4][0] for i in infos],
        "connect_attempts": attempts,
    }


def time_calls(url: str, n: int, body: bytes | None = None) -> list[float]:
    out = []
    for _ in range(n):
        if body is None:
            request = urllib.request.Request(url)
        else:
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                response.read()
        except urllib.error.URLError as exc:
            raise SystemExit(f"{url} is not answering: {exc}") from exc
        out.append((time.perf_counter() - started) * 1000)
    return out


def stats(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "median_ms": round(float(np.median(xs)), 1),
        "mean_ms": round(float(np.mean(xs)), 1),
        "min_ms": round(float(np.min(xs)), 1),
        "max_ms": round(float(np.max(xs)), 1),
    }


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()
    llm = config["llm"]
    port = int(llm.get("base_url", "http://127.0.0.1:11434").rsplit(":", 1)[1])

    body = json.dumps({
        "model": llm["model"], "prompt": PROMPT, "format": "json", "stream": False,
        "options": {"temperature": llm.get("temperature", 0.0),
                    "num_predict": llm.get("max_tokens", 64)},
    }).encode()

    # Warm the model so the first call does not pay for loading weights; that is
    # a real cost but a one-off, and it belongs to neither arm.
    log.info("warming the model ...")
    time_calls(f"http://127.0.0.1:{port}/api/generate", 1, body)

    report: dict = {"resolution": [], "no_work": {}, "generate": {}}
    for host in HOSTS:
        report["resolution"].append(resolution(host, port))

        idle = time_calls(f"http://{host}:{port}/api/tags", args.n)
        report["no_work"][host] = stats(idle)
        log.info("%-9s /api/tags   median %8.1f ms", host, np.median(idle))

        gen = time_calls(f"http://{host}:{port}/api/generate", args.n, body)
        report["generate"][host] = stats(gen)
        log.info("%-9s /api/generate median %8.1f ms", host, np.median(gen))

    fast, slow = "127.0.0.1", "localhost"
    client_overhead = (report["no_work"][slow]["median_ms"]
                       - report["no_work"][fast]["median_ms"])
    model_cost = report["generate"][fast]["median_ms"]
    hostname_cost = report["generate"][slow]["median_ms"]

    report["conclusion"] = {
        "fixed_client_overhead_ms": round(client_overhead, 1),
        "model_median_ms": round(model_cost, 1),
        "as_measured_over_localhost_ms": round(hostname_cost, 1),
        "share_of_localhost_latency_that_was_the_client": round(
            client_overhead / hostname_cost, 4),
        "cause": ("localhost resolves to ::1 first; Ollama binds IPv4 only, so "
                  "urllib waits for the IPv6 connection to be refused before "
                  "falling back to 127.0.0.1, on every call"),
        "note": ("This is a property of this machine's loopback configuration, "
                 "not of Ollama or of the model. On a host where localhost "
                 "resolves to 127.0.0.1 first, or where Ollama listens on ::1, "
                 "the overhead is absent. That is exactly why it must not be "
                 "inside a published per-tender latency."),
    }

    save_json(report, "reports/metrics_llm_overhead.json")

    print()
    print("| Host | /api/tags (no work) | /api/generate | ")
    print("|---|---:|---:|")
    for host in HOSTS:
        print(f"| {host} | {report['no_work'][host]['median_ms']:.1f} ms | "
              f"{report['generate'][host]['median_ms']:.1f} ms |")
    print()
    print(f"Fixed client overhead: {client_overhead:.0f} ms per call "
          f"({100 * client_overhead / hostname_cost:.0f}% of a generate call "
          f"over the hostname).")
    print(f"The model itself: {model_cost:.0f} ms median per tender.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Lightweight local web UI server for the multimodal fusion-transformer trainer.

Serves the training console in ``ui/index.html`` and exposes a JSON API that runs the real
``Pipeline``. Standard library only. Start with ``python ui/server.py`` and open the
printed URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")


def run_training(
    num_epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    num_samples: int = 600,
    num_features: int = 8,
    d_model: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a fusion-transformer on synthetic multimodal data and return metrics.

    Raises ValueError with a clear message for invalid parameters, matching the CLI's
    boundary validation.
    """
    for name, value in (
        ("num_epochs", num_epochs),
        ("batch_size", batch_size),
        ("num_samples", num_samples),
        ("num_features", num_features),
        ("d_model", d_model),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
    if num_samples < 50:
        raise ValueError(f"num_samples must be >= 50, got {num_samples}")
    if num_features < 4:
        raise ValueError(f"num_features must be >= 4, got {num_features}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    config = Config(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_samples=num_samples,
        num_features=num_features,
        d_model=d_model,
        seed=seed,
    )
    pipeline = Pipeline(config)
    summary = pipeline.train()

    def _round(values):
        return [round(float(v), 6) for v in values]

    return {
        "config": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_samples": num_samples,
            "num_features": num_features,
            "d_model": d_model,
            "seed": seed,
        },
        "train_losses": _round(summary.train_losses),
        "val_losses": _round(summary.val_losses),
        "val_metrics": {k: round(float(v), 6) for k, v in summary.val_metrics.items()},
        "test_metrics": {k: round(float(v), 6) for k, v in summary.test_metrics.items()},
    }


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok"})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/train":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = run_training(
                    num_epochs=payload.get("num_epochs", 8),
                    batch_size=payload.get("batch_size", 32),
                    learning_rate=payload.get("learning_rate", 2e-3),
                    num_samples=payload.get("num_samples", 600),
                    num_features=payload.get("num_features", 8),
                    d_model=payload.get("d_model", 32),
                    seed=payload.get("seed", 42),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the fusion-transformer training console UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8130)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"Fusion-transformer training console running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
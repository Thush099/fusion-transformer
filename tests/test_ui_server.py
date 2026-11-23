"""Tests for the fusion-transformer training console UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_training

FAST = {"num_epochs": 1, "num_samples": 120, "num_features": 6, "d_model": 16, "batch_size": 16}


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/train",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_train_endpoint_runs_real_pipeline(server):
    status, payload = _post(server, FAST)
    assert status == 200
    assert len(payload["train_losses"]) == 1
    assert len(payload["val_losses"]) == 1
    for key in ("accuracy", "macro_f1", "precision", "recall", "roc_auc"):
        assert key in payload["test_metrics"]


def test_result_is_json_serializable():
    result = run_training(**FAST)
    json.dumps(result)  # must not raise


def test_health_and_index_served(server):
    assert json.loads(urllib.request.urlopen(server + "/api/health").read())["status"] == "ok"
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "training console" in html and "/api/train" in html


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"num_epochs": "x"}, "num_epochs must be an integer"),
        ({"num_epochs": 0}, "num_epochs must be >= 1"),
        ({"num_samples": 10}, "num_samples must be >= 50"),
        ({"num_features": 2}, "num_features must be >= 4"),
        ({"learning_rate": 0}, "learning_rate must be > 0"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    merged = {**FAST, **payload}
    status, body = _post(server, merged)
    assert status == 400
    assert message in body["error"]
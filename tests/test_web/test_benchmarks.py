import json
import tempfile
from pathlib import Path

from redis_guardrails.models import Match


def test_index_lists_preset_files(client, tmp_path, monkeypatch):
    (tmp_path / "sample.json").write_text("[]")
    monkeypatch.setattr("redis_guardrails.web.routes.benchmarks.DATA_DIR", tmp_path)

    response = client.get("/benchmarks")
    assert response.status_code == 200
    assert "sample.json" in response.text


def test_run_with_preset_shows_pass_fail_and_summary(client, store, tmp_path, monkeypatch):
    monkeypatch.setattr("redis_guardrails.web.routes.benchmarks.DATA_DIR", tmp_path)
    store.matches_by_text["bad text"] = [
        Match(rule_id="g-1", category="cat", action="BLOCK", distance=0.1, threshold=0.5, chunk_id="input-0", evaluated_text="bad text")
    ]
    (tmp_path / "sample.json").write_text(json.dumps([
        {"id": "case-1", "stage": "input", "input": "bad text", "category": "cat", "action": "BLOCK"},
    ]))

    response = client.post("/benchmarks/run", data={"source": "preset", "preset_path": "sample.json"})
    assert response.status_code == 200
    assert "case-1" in response.text
    assert "PASS" in response.text
    assert "Evaluations:" in response.text
    assert "Avg total time" in response.text


def test_run_with_uploaded_file_success(client):
    payload = json.dumps([
        {"id": "case-1", "stage": "input", "input": "anything", "category": "cat", "action": "ALLOW"},
    ]).encode()
    response = client.post(
        "/benchmarks/run",
        data={"source": "upload"},
        files={"upload_file": ("cases.json", payload, "application/json")},
    )
    assert response.status_code == 200
    assert "case-1" in response.text


def test_run_with_uploaded_file_cleans_up_tempfile(client, monkeypatch):
    created_paths: list[Path] = []
    original_named_temp_file = tempfile.NamedTemporaryFile

    def capturing_named_temp_file(*args, **kwargs):
        tmp = original_named_temp_file(*args, **kwargs)
        created_paths.append(Path(tmp.name))
        return tmp

    monkeypatch.setattr(
        "redis_guardrails.web.routes.benchmarks.tempfile.NamedTemporaryFile",
        capturing_named_temp_file,
    )

    payload = json.dumps([
        {"id": "case-1", "stage": "input", "input": "anything", "category": "cat", "action": "ALLOW"},
    ]).encode()
    response = client.post(
        "/benchmarks/run",
        data={"source": "upload"},
        files={"upload_file": ("cases.json", payload, "application/json")},
    )
    assert response.status_code == 200
    assert len(created_paths) == 1
    assert not created_paths[0].exists()


def test_run_with_invalid_chunking_override_shows_inline_error_not_traceback(client, tmp_path, monkeypatch):
    monkeypatch.setattr("redis_guardrails.web.routes.benchmarks.DATA_DIR", tmp_path)
    (tmp_path / "sample.json").write_text(json.dumps([
        {"id": "case-1", "stage": "input", "input": "anything", "category": "cat", "action": "ALLOW"},
    ]))

    response = client.post(
        "/benchmarks/run",
        data={"source": "preset", "preset_path": "sample.json", "max_chars": "abc"},
    )
    assert response.status_code == 400
    assert "Traceback" not in response.text


def test_run_with_malformed_json_shows_inline_error_not_traceback(client):
    response = client.post(
        "/benchmarks/run",
        data={"source": "upload"},
        files={"upload_file": ("bad.json", b"not json", "application/json")},
    )
    assert response.status_code == 400
    assert "Traceback" not in response.text


def test_run_rejects_unknown_preset_path(client, tmp_path, monkeypatch):
    monkeypatch.setattr("redis_guardrails.web.routes.benchmarks.DATA_DIR", tmp_path)
    response = client.post("/benchmarks/run", data={"source": "preset", "preset_path": "../../etc/passwd"})
    assert response.status_code == 400

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from autolabel_lab.cancellation import check_cancelled
from autolabel_lab.project import LabError, Project, write_json
from autolabel_lab.web import Workspace, command_args, contained, curve, make_server


@pytest.fixture
def server(tmp_path):
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, path, payload=None, token=None, host=None, origin=None):
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
        if token:
            headers["X-Lab-Token"] = token
    if host:
        headers["Host"] = host
    if origin:
        headers["Origin"] = origin
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}",
                                 data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def test_bootstrap_and_create_project(server):
    boot = request(server, "/api/bootstrap")
    assert boot["app"] == "autolabel-lab"
    assert boot["projects"] == []
    created = request(server, "/api/create", {"name": "demo", "title": "旋转目标试验", "classes": ["目标"]}, boot["token"])
    details = request(server, "/api/project?id=" + created["id"])
    assert details["config"]["model_yaml"] == "yolo26s-obb.yaml"
    assert details["summary"]["frames"]["seed"] == 0
    assert details["models"] == []


def test_mutation_requires_token_and_local_origin(server):
    with pytest.raises(urllib.error.HTTPError) as result:
        request(server, "/api/create", {"name": "demo", "classes": ["a"]})
    assert result.value.code == 403
    token = request(server, "/api/bootstrap")["token"]
    with pytest.raises(urllib.error.HTTPError):
        request(server, "/api/create", {"name": "demo", "classes": ["a"]}, token, origin="https://untrusted.example")
    with pytest.raises(urllib.error.HTTPError):
        request(server, "/api/bootstrap", host="untrusted.example")
    assert not (server.workspace.root / "experiments" / "demo").exists()


def test_file_access_stays_inside_registered_experiment(server, tmp_path):
    token = request(server, "/api/bootstrap")["token"]
    created = request(server, "/api/create", {"name": "demo", "classes": ["a"]}, token)
    (tmp_path / "outside.json").write_text('{"secret":1}', encoding="utf-8")
    with pytest.raises(urllib.error.HTTPError):
        request(server, f"/api/file?id={created['id']}&path=../../outside.json")
    assert request(server, f"/api/file?id={created['id']}&path=project.json")["classes"] == ["a"]


def test_command_uses_fixed_flags_without_shell_interpretation():
    args = command_args("extract", {"video": "D:/path/$()/video.mp4", "every": 1})
    assert args == ["extract", "--video=D:/path/$()/video.mp4", "--every=1"]
    assert command_args("train", {"name": "M0", "rounds": ["r1", "r2"], "dry_run": True})[-1] == "--dry-run"
    with pytest.raises(LabError):
        command_args("train", {"unknown": "x"})
    with pytest.raises(LabError):
        command_args("delete", {})


def test_job_executes_cli_and_surfaces_failure(server):
    project = Project.create(server.workspace.root / "experiments" / "demo", ["a"])
    entry = server.workspace.register(project.root)
    job = server.workspace.start_job(entry["id"], "split", {"seed_count": 15})
    deadline = time.monotonic() + 15
    while job["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(.05)
    assert job["status"] == "failed"
    assert "请先抽帧" in server.workspace.job_log(job["id"])
    assert not (project.root / ".operation.lock").exists()


def test_training_curve_parses_real_headers_and_partial_lines(tmp_path):
    path = tmp_path / "runs" / "train" / "results.csv"
    path.parent.mkdir(parents=True)
    path.write_text(" epoch, train/box_loss, val/box_loss\n1,2.5,3.2\n2,2.0,\n", encoding="utf-8")
    rows = curve(tmp_path)
    assert rows[0]["train/box_loss"] == 2.5
    assert rows[1]["epoch"] == 2
    assert "val/box_loss" not in rows[1]


def test_restart_preserves_projects_and_marks_interrupted_jobs(tmp_path):
    workspace = Workspace(tmp_path)
    p = Project.create(tmp_path / "experiments" / "demo", ["a"])
    workspace.register(p.root)
    write_json(workspace.state_dir / "jobs" / "old" / "job.json", {"id": "old", "status": "running", "action": "train"})
    reopened = Workspace(tmp_path)
    assert len(reopened.state["projects"]) == 1
    assert reopened.jobs["old"]["status"] == "interrupted"


def test_cancel_marker_is_cooperative_and_does_not_delete_data(tmp_path, monkeypatch):
    marker = tmp_path / "cancel"
    monkeypatch.setenv("ALAB_CANCEL_FILE", str(marker))
    check_cancelled()
    marker.touch()
    with pytest.raises(LabError, match="停止"):
        check_cancelled()
    assert marker.exists()


def test_details_orders_models_by_creation_time_not_name(tmp_path):
    workspace = Workspace(tmp_path)
    project = Project.create(tmp_path / "experiments" / "demo", ["a"])
    entry = workspace.register(project.root)
    for name, created in [("M2", "2026-01-01"), ("M10", "2026-01-02")]:
        write_json(project.root / "models" / name / "model.json",
                   {"name": name, "created_at": created, "init": "scratch", "status": "completed", "counts": {"train": 1}})
    assert [m["name"] for m in workspace.details(entry["id"])["models"]] == ["M2", "M10"]


def test_concurrent_registration_preserves_registry(tmp_path):
    workspace = Workspace(tmp_path)
    projects = [Project.create(tmp_path / "experiments" / f"p{i}", ["a"]) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda p: workspace.register(p.root), projects * 2))
    saved = json.loads(workspace.registry_file.read_text(encoding="utf-8"))
    assert len(saved["projects"]) == 4
    assert {p["path"] for p in saved["projects"]} == {str(p.root) for p in projects}


def test_invalid_json_shape_returns_api_error(server):
    token = request(server, "/api/bootstrap")["token"]
    with pytest.raises(urllib.error.HTTPError) as result:
        request(server, "/api/create", [], token)
    assert result.value.code == 400
    assert "JSON" in json.loads(result.value.read())["error"]

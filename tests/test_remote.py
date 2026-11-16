import tarfile
from pathlib import Path

import pytest

from autolabel_lab.project import LabError, read_json, sha256, write_json
from autolabel_lab.remote import validate_devices, train_remote, monitor, runner_command, server_config
from test_image_import import source, prepare


def test_gpu_selection_and_quoted_commands():
    validate_devices("0", 3)
    validate_devices("0,1", 4)
    with pytest.raises(LabError, match="倍数"):
        validate_devices("0,1", 3)
    with pytest.raises(LabError):
        validate_devices("0;echo secret", 4)
    state = {"server": {"python": "/home/user name/env/bin/python"}, "job_path": "/home/user name/jobs/job"}
    import shlex
    assert shlex.split(runner_command(state, "status", "--offset", "2"))[0] == state["server"]["python"]


def test_remote_transfer_excludes_test_and_round_reference_answers(tmp_path, source, monkeypatch):
    from autolabel_lab.dataset import build_dataset
    project, _ = prepare(tmp_path, source)
    dataset, snapshot = build_dataset(project, "M0", [])
    root = project.root / "models/M0"
    server = {"id": "test", "host": "server", "python": "/python", "root": "/jobs"}
    record = {"name": "M0", "init": "scratch", "model_yaml": "yolo26s-obb.yaml", "params": {"batch": 4, "device": "0", "seed": 42}}
    monkeypatch.setattr("autolabel_lab.remote.probe_server", lambda s: {"ready": True, "python_version": "3.10", "torch": "2.5.1", "ultralytics": "8.4.91", "cuda": "12.4"})
    monkeypatch.setattr("autolabel_lab.remote.ssh", lambda *a, **k: "")
    monkeypatch.setattr("autolabel_lab.remote.scp", lambda *a, **k: None)
    def fake_monitor(project, root, state, record):
        with tarfile.open(root / "transfer.tar.gz") as archive:
            names = archive.getnames()
        assert any(n.startswith("dataset/images/train/") for n in names)
        assert any(n.startswith("dataset/images/val/") for n in names)
        assert not any("/test/" in n or "references/" in n for n in names)
        assert "dataset/data.yaml" not in names
        request = read_json(root / "request.json")
        assert set(request["transfer_hashes"]) == {n for n in names if n.startswith("dataset/")}
        record["status"] = "completed"
        write_json(root / "model.json", record)
    monkeypatch.setattr("autolabel_lab.remote.monitor", fake_monitor)
    assert train_remote(project, root, dataset, snapshot, record, server)["status"] == "completed"


def test_monitor_verifies_returned_checkpoint_and_updates_live_curve(tmp_path, monkeypatch):
    root = tmp_path / "models/M0"
    root.mkdir(parents=True)
    class Project:
        pass
    project = Project()
    project.root = tmp_path
    payload = b"remote verified checkpoint"
    original = tmp_path / "expected.pt"
    original.write_bytes(payload)
    state = {"server": {"host": "server", "python": "/python"}, "job_path": "/jobs/test", "token": "a", "offset": 0, "phase": "running"}
    record = {"status": "running"}
    import json
    monkeypatch.setattr("autolabel_lab.remote.ssh", lambda *a, **k: json.dumps({"log": "epoch 1\n", "offset": 8,
        "csv": "epoch,train/box_loss\n1,2.0\n", "alive": False, "result": {"token": "a", "status": "completed",
            "weights_sha256": sha256(original), "machine_seconds": 4, "runtime": {"torch": "2.5.1"}}}))
    monkeypatch.setattr("autolabel_lab.remote.scp", lambda s, src, dst: Path(dst).write_bytes(payload))
    monitor(project, root, state, record)
    assert (root / "best.pt").read_bytes() == payload
    assert read_json(root / "model.json")["status"] == "completed"
    assert "1,2.0" in (root / "runs/train/results.csv").read_text()


def test_connection_uncertainty_keeps_remote_job_recoverable(tmp_path, source, monkeypatch):
    from autolabel_lab.dataset import build_dataset
    project, _ = prepare(tmp_path, source)
    dataset, snapshot = build_dataset(project, "M0", [])
    server = {"id": "test", "host": "server", "python": "/python", "root": "/jobs"}
    record = {"name": "M0", "init": "scratch", "model_yaml": "yolo26s-obb.yaml", "params": {"batch": 4, "device": "0"}}
    monkeypatch.setattr("autolabel_lab.remote.probe_server", lambda s: {"ready": True, "python_version": "3.10", "torch": "2.5.1", "ultralytics": "8.4.91", "cuda": "12.4"})
    monkeypatch.setattr("autolabel_lab.remote.ssh", lambda *a, **k: "")
    monkeypatch.setattr("autolabel_lab.remote.scp", lambda *a, **k: None)
    monkeypatch.setattr("autolabel_lab.remote.monitor", lambda *a: (_ for _ in ()).throw(LabError("connection lost")))
    root = project.root / "models/M0"
    with pytest.raises(LabError):
        train_remote(project, root, dataset, snapshot, record, server)
    assert read_json(root / "model.json")["status"] == "remote_disconnected"
    assert read_json(root / "remote.json")["phase"] == "launching"


def test_workspace_blocks_new_work_until_uncertain_remote_job_is_recovered(tmp_path):
    from autolabel_lab.project import Project
    from autolabel_lab.web import Workspace
    workspace = Workspace(tmp_path)
    project = Project.create(tmp_path / "experiments/demo", ["target"])
    entry = workspace.register(project.root)
    write_json(project.root / "models/M0/remote.json", {"phase": "running"})
    write_json(project.root / "models/M0/model.json", {"status": "remote_disconnected"})
    with pytest.raises(LabError, match="恢复远程任务"):
        workspace.start_job(entry["id"], "report", {})
    assert workspace.jobs == {}


def test_server_config_is_local_and_not_secret_material(tmp_path):
    write_json(tmp_path / "servers.local.json", {"servers": [{"id": "custom", "host": "172.21.23.182", "port": 22}]})
    assert server_config(tmp_path, "custom")["host"] == "172.21.23.182"
    with pytest.raises(LabError):
        server_config(tmp_path, "missing")

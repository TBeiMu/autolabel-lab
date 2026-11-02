"""Exercise a real parent-process exit and recover the independent CLI worker."""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil
import pytest

from autolabel_lab.project import LabError, Project, write_json
from autolabel_lab.web import Workspace


def wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.05)
    raise AssertionError("Worker did not reach expected state")


def test_recover_cancel_and_collect_result_after_server_process_exits(tmp_path):
    source = str(Path(__file__).resolve().parents[1] / "src")
    worker = tmp_path / "slow_worker.py"
    worker.write_text('''import sys, time
from pathlib import Path
from autolabel_lab import cli, job_worker
from autolabel_lab.cancellation import check_cancelled
from autolabel_lab.project import Project
def slow_cli(argv):
    project = Project(argv[argv.index('--project')+1])
    with project.lock():
        while True:
            check_cancelled()
            time.sleep(.05)
cli.main = slow_cli
raise SystemExit(job_worker.main())
''', encoding="utf-8")
    launcher = tmp_path / "server.py"
    launcher.write_text('''import json, os, subprocess, sys
from pathlib import Path
from autolabel_lab.project import Project
from autolabel_lab.web import make_server
root=Path(os.environ['TEST_WORKSPACE'])
real_popen=subprocess.Popen
def spawn_worker(argv, **kwargs):
    return real_popen([sys.executable, '-u', str(root/'slow_worker.py'), *argv[4:]], **kwargs)
subprocess.Popen=spawn_worker
server=make_server(root, port=0)
first=server.workspace.register(Project.create(root/'experiments'/'a', ['target']).root)
second=server.workspace.register(Project.create(root/'experiments'/'b', ['target']).root)
(root/'ready.json').write_text(json.dumps({'port': server.server_port, 'token': server.workspace.token, 'first':first, 'second':second}))
server.serve_forever()
''', encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": source, "TEST_WORKSPACE": str(tmp_path)}
    parent = subprocess.Popen([sys.executable, str(launcher)], env=env,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    child = None
    try:
        wait_until(lambda: (tmp_path / "ready.json").is_file())
        ready = json.loads((tmp_path / "ready.json").read_text())
        request = urllib.request.Request(f"http://127.0.0.1:{ready['port']}/api/job",
                                         data=json.dumps({"project_id": ready["first"]["id"], "action": "train", "payload": {"name": "M0"}}).encode(),
                                         headers={"Content-Type": "application/json", "X-Lab-Token": ready["token"]})
        with urllib.request.urlopen(request, timeout=10) as response:
            job = json.loads(response.read())
        folder = tmp_path / ".gui" / "jobs" / job["id"]
        wait_until(lambda: (folder / "worker.json").is_file() and (tmp_path / "experiments" / "a" / ".operation.lock").exists())
        identity = json.loads((folder / "worker.json").read_text())
        child = psutil.Process(identity["pid"])
        parent.kill()
        parent.wait(timeout=10)
        assert child.is_running()
        recovered = Workspace(tmp_path)
        assert recovered.jobs[job["id"]]["status"] == "running"
        with pytest.raises(LabError, match="已有任务"):
            recovered.start_job(ready["second"]["id"], "train", {"name": "M1"})
        assert recovered.cancel_job(job["id"])["status"] == "cancelling"
        wait_until(lambda: (folder / "result.json").is_file())
        jobs = recovered.overview()["jobs"]
        assert next(j for j in jobs if j["id"] == job["id"])["status"] == "cancelled"
        assert not (tmp_path / "experiments" / "a" / ".operation.lock").exists()
        # The cancellation is durable through another restart.
        assert Workspace(tmp_path).jobs[job["id"]]["status"] == "cancelled"
        next_job = recovered.start_job(ready["second"]["id"], "split", {"seed_count": 1})
        wait_until(lambda: next_job["status"] == "failed")
        assert "请先抽帧" in recovered.job_log(next_job["id"])
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        if child:
            try:
                child.wait(timeout=2)
            except psutil.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def test_reused_pid_cannot_be_adopted_or_block_new_jobs(tmp_path):
    project = Project.create(tmp_path / "experiments" / "a", ["target"])
    workspace = Workspace(tmp_path)
    entry = workspace.register(project.root)
    process = psutil.Process()
    write_json(workspace.state_dir / "jobs" / "old" / "job.json",
               {"id": "old", "project_id": entry["id"], "action": "train", "payload": {"name": "M0"},
                "status": "running", "worker_token": "original", "created_at": "2020-01-01T00:00:00+00:00",
                "worker": {"pid": process.pid, "create_time": process.create_time() - 100}})
    recovered = Workspace(tmp_path)
    assert recovered.jobs["old"]["status"] == "interrupted"
    assert process.is_running()


def test_completion_written_while_server_offline_is_recovered(tmp_path):
    workspace = Workspace(tmp_path)
    write_json(workspace.state_dir / "jobs" / "done" / "job.json",
               {"id": "done", "status": "running", "worker_token": "original", "action": "train"})
    write_json(workspace.state_dir / "jobs" / "done" / "result.json",
               {"token": "original", "exit_code": 0, "completed_at": "2026-01-01T00:00:00+00:00"})
    recovered = Workspace(tmp_path)
    assert recovered.jobs["done"]["status"] == "completed"
    assert recovered.jobs["done"]["exit_code"] == 0

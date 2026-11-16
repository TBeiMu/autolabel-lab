"""SSH transport and durable local monitor for dedicated GPU servers."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tarfile
import time
import uuid
from pathlib import Path, PurePosixPath

from .project import LabError, now, read_json, sha256, write_json

DEFAULT_SERVERS = [
    {"id": "gpu205", "host": "172.21.23.205", "label": "172.21.23.205 · 2 × RTX 4090 D", "python": "", "root": "", "port": 22},
    {"id": "gpu182", "host": "172.21.23.182", "label": "172.21.23.182 · 2 × RTX 3090", "python": "", "root": "", "port": 22},
]


def servers(workspace):
    path = Path(workspace) / "servers.local.json"
    data = read_json(path) if path.exists() else {"servers": DEFAULT_SERVERS}
    return data["servers"]


def server_config(workspace, server_id):
    server = next((s for s in servers(workspace) if s["id"] == server_id), None)
    if not server:
        raise LabError("服务器配置不存在。")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", server["host"]) or server["host"].startswith("-"):
        raise LabError("服务器地址格式不正确。")
    if not 1 <= int(server.get("port", 22)) <= 65535:
        raise LabError("SSH 端口不正确。")
    for key in ["python", "root"]:
        if server.get(key) and (not str(server[key]).startswith("/") or any(c in str(server[key]) for c in "\r\n\0")):
            raise LabError("服务器 Python 和工作目录需要 Linux 绝对路径。")
    return server


def ssh(server, command, timeout=30):
    args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", "-p", str(server.get("port", 22)), server["host"], command]
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LabError(f"SSH 连接失败：{exc}") from exc
    if result.returncode:
        raise LabError(f"服务器 {server['host']}：{result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def scp(server, source, destination):
    result = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                             "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10", "-P", str(server.get("port", 22)),
                             str(source), str(destination)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, timeout=1800)
    if result.returncode:
        raise LabError(f"传输失败：{result.stderr.strip()}")


def remote_path(server, path):
    # OpenSSH's SFTP-based scp accepts a literal absolute remote path.
    return f"{server['host']}:{path}"


def probe_server(server):
    if not server.get("python") or not server.get("root"):
        raise LabError("服务器训练环境尚未配置。")
    source = (Path(__file__).parent / "remote_runner.py").read_text(encoding="utf-8")
    command = shlex.join([server["python"], "-c", source, "probe"])
    result = json.loads(ssh(server, command, timeout=30))
    result["ready"] = result["ultralytics"] == "8.4.91" and result["cuda_available"]
    return result


def public_servers(workspace):
    result = []
    for server in servers(workspace):
        entry = {key: server.get(key) for key in ["id", "host", "label", "python", "root", "port"]}
        entry["configured"] = bool(server.get("python") and server.get("root"))
        result.append(entry)
    return result


def validate_devices(device, batch):
    if device not in {"0", "1", "0,1"}:
        raise LabError("服务器 GPU 请选择 0、1 或 0,1。")
    if device == "0,1" and batch % 2:
        raise LabError("双 GPU 的总 batch 必须是 2 的倍数。")


def train_remote(project, root, dataset, snapshot, record, server, parent_weights=None):
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    validate_devices(record["params"]["device"], record["params"]["batch"])
    root.mkdir(parents=True)
    record.update(status="running", execution={"target": "server", "host": server["host"], "server_id": server["id"]})
    started = time.monotonic()
    token = uuid.uuid4().hex
    job_path = str(PurePosixPath(server["root"]) / "jobs" / token)
    state_path = root / "remote.json"
    state = {"server": server, "job_path": job_path, "token": token, "offset": 0, "phase": "preparing"}
    write_json(root / "model.json", record)
    write_json(state_path, state)
    try:
        info = probe_server(server)
        if not info["ready"]:
            raise LabError("服务器环境没有可用 CUDA 或 Ultralytics 版本不是 8.4.91。")
        record["versions"] = {"python": info["python_version"], "torch": info["torch"], "ultralytics": info["ultralytics"], "cuda": info["cuda"], "platform": "Linux"}
        hashes = {}
        bundle = root / "transfer.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            for relative, digest in snapshot["file_hashes"].items():
                if relative.startswith(("images/train/", "images/val/", "labels/train/", "labels/val/")):
                    if sha256(dataset / relative) != digest:
                        raise LabError("传输前数据快照发生变化。")
                    name = "dataset/" + relative
                    archive.add(dataset / relative, arcname=name)
                    hashes[name] = digest
            for path in (Path(__file__).parent).glob("*.py"):
                archive.add(path, arcname="code/autolabel_lab/" + path.name)
            remote_model = record["model_yaml"]
            if record["init"] == "continue":
                archive.add(parent_weights, arcname="parent.pt")
                hashes["parent.pt"] = record["parent_weights_sha256"]
                remote_model = str(PurePosixPath(job_path) / "parent.pt")
            elif Path(remote_model).is_file():
                archive.add(remote_model, arcname="structure.yaml")
                hashes["structure.yaml"] = sha256(remote_model)
                remote_model = str(PurePosixPath(job_path) / "structure.yaml")
        request = {"token": token, "init": record["init"], "remote_model": remote_model, "params": record["params"],
                   "classes": project.config["classes"], "transfer_hashes": hashes}
        write_json(root / "request.json", request)
        print(f"传输训练/验证数据到 {server['host']}；固定测试集不上传。", flush=True)
        ssh(server, shlex.join(["mkdir", "-p", job_path]))
        scp(server, bundle, remote_path(server, job_path + "/transfer.tar.gz"))
        scp(server, root / "request.json", remote_path(server, job_path + "/request.json"))
        ssh(server, shlex.join(["tar", "-xzf", job_path + "/transfer.tar.gz", "-C", job_path]))
        state["phase"] = "launching"
        write_json(state_path, state)
        record["execution"].update(job_path=job_path, device=record["params"]["device"])
        write_json(root / "model.json", record)
        monitor(project, root, state, record)
        return read_json(root / "model.json")
    except BaseException as error:
        # An uncertain SSH launch must remain recoverable; never offer a second training
        # while a detached remote job may still be running.
        if state["phase"] in {"launching", "running"}:
            record.update(status="remote_disconnected", error=str(error))
        else:
            record.update(status="failed", error=str(error))
        record.setdefault("machine_seconds", time.monotonic() - started)
        write_json(root / "model.json", record)
        raise


def runner_command(state, action, *extra):
    return shlex.join([state["server"]["python"], state["job_path"] + "/code/autolabel_lab/remote_runner.py",
                       action, state["job_path"], *extra])


def monitor(project, root, state, record):
    cancel_sent = False
    disconnected_at = None
    while True:
        marker = os.environ.get("ALAB_CANCEL_FILE")
        try:
            if marker and Path(marker).exists() and not cancel_sent:
                ssh(state["server"], runner_command(state, "cancel"))
                cancel_sent = True
            if state["phase"] == "launching":
                identity = json.loads(ssh(state["server"], runner_command(state, "launch")))
                if identity["token"] != state["token"]:
                    raise LabError("远程进程身份不一致。")
                state["phase"] = "running"
                state["identity"] = identity
                write_json(root / "remote.json", state)
            update = json.loads(ssh(state["server"], runner_command(state, "status", "--offset", str(state["offset"]))))
            disconnected_at = None
            if update["log"]:
                print(update["log"], end="", flush=True)
            state["offset"] = update["offset"]
            write_json(root / "remote.json", state)
            if update["csv"]:
                path = root / "runs/train/results.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".tmp")
                temp.write_text(update["csv"], encoding="utf-8")
                temp.replace(path)
            result = update["result"]
            if result:
                if result["token"] != state["token"]:
                    raise LabError("远程结果与训练任务不一致。")
                if result["status"] != "completed":
                    record.update(status="failed", error=result.get("error", "远程训练失败。"), machine_seconds=result["machine_seconds"])
                    write_json(root / "model.json", record)
                    state["phase"] = "finished"
                    write_json(root / "remote.json", state)
                    raise LabError(record["error"])
                destination = root / "best.pt.part"
                scp(state["server"], remote_path(state["server"], state["job_path"] + "/runs/train/weights/best.pt"), destination)
                if sha256(destination) != result["weights_sha256"]:
                    raise LabError("回传权重哈希不一致。")
                destination.replace(root / "best.pt")
                record.update(status="completed", completed_at=now(), weights=(root / "best.pt").relative_to(project.root).as_posix(),
                              weights_sha256=result["weights_sha256"], machine_seconds=result["machine_seconds"],
                              remote_runtime=result["runtime"])
                record.pop("error", None)
                write_json(root / "model.json", record)
                state["phase"] = "finished"
                write_json(root / "remote.json", state)
                print("服务器训练完成，best.pt 已回传并核验。", flush=True)
                return
            if not update["alive"]:
                state["phase"] = "finished"
                write_json(root / "remote.json", state)
                record.update(status="failed", error="远程训练进程已退出，但未保存完整结果。")
                write_json(root / "model.json", record)
                raise LabError("远程训练进程已退出，但未保存完整结果。")
        except (LabError, subprocess.TimeoutExpired) as error:
            if state["phase"] == "finished":
                raise
            disconnected_at = disconnected_at or time.monotonic()
            print(f"远程连接暂不可用，保留任务并重试：{error}", flush=True)
            if time.monotonic() - disconnected_at > 120:
                raise LabError("服务器连接中断，远程任务可能仍在运行；请使用恢复远程任务重新接管。") from error
        time.sleep(3)


def recover(project, name, workspace):
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from .project import safe_name
    root = project.root / "models" / safe_name(name)
    state = read_json(root / "remote.json")
    record = read_json(root / "model.json")
    if state["phase"] not in {"launching", "running"} or record["status"] == "completed":
        raise LabError("此模型没有待恢复的远程任务。")
    state["server"] = server_config(workspace, state["server"]["id"])
    record["status"] = "running"
    write_json(root / "model.json", record)
    try:
        monitor(project, root, state, record)
    except BaseException as error:
        record.update(status="remote_disconnected" if state["phase"] != "finished" else "failed", error=str(error))
        write_json(root / "model.json", record)
        raise

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import psutil

from .project import DEFAULT_MODEL_YAML, LabError, Project, now, read_json, safe_name, write_json
from .reports import status

STATIC = Path(__file__).parent / "static"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise LabError("路径超出实验目录。")
    return path


def finite_number(value, name, integer=False):
    try:
        number = float(value)
        if not np.isfinite(number) or (integer and int(number) != number):
            raise ValueError()
        return int(number) if integer else number
    except (TypeError, ValueError, OverflowError) as exc:
        raise LabError(f"{name} 必须是{'整数' if integer else '有限数字'}。") from exc


def curve(model_root):
    paths = list((model_root / "runs").glob("*/results.csv"))
    if not paths:
        return []
    try:
        rows = []
        with paths[0].open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                parsed = {}
                for key, value in row.items():
                    if key and value:
                        try:
                            number = float(value.strip())
                            if np.isfinite(number):
                                parsed[key.strip()] = number
                        except ValueError:
                            pass
                if "epoch" in parsed:
                    rows.append(parsed)
        return rows
    except OSError:
        return []


def command_args(action, payload):
    """Build a shell-free CLI call from a fixed set of supported operations."""
    fields = {
        "extract": {"video": "--video", "every": "--every", "start": "--start", "end": "--end", "max_frames": "--max-frames", "timestamps": "--timestamps"},
        "split": {"seed_count": "--seed-count", "val_fraction": "--val-fraction", "test_fraction": "--test-fraction", "gap_seconds": "--gap-seconds"},
        "import-yolo": {"source": "--source", "train_count": "--train-count", "val_count": "--val-count", "pool_count": "--pool-count", "test_count": "--test-count"},
        "import-labels": {"subset": "--subset", "round": "--round", "source": "--from", "reviewed": "--reviewed"},
        "train": {"name": "--name", "rounds": "--rounds", "init": "--init", "from_model": "--from-model", "epochs": "--epochs", "imgsz": "--imgsz", "batch": "--batch", "device": "--device", "dry_run": "--dry-run"},
        "autolabel": {"round": "--round", "model": "--model", "count": "--count", "conf": "--conf", "device": "--device"},
        "evaluate": {"models": "--models", "device": "--device"},
        "time": {"stage": "--stage", "seconds": "--seconds", "round": "--round", "note": "--note"},
        "report": {},
    }
    if action not in fields or set(payload) - set(fields[action]):
        raise LabError("不支持的操作或参数。")
    args = [action]
    for key, value in payload.items():
        if value is None or value == "" or value == []:
            continue
        flag = fields[action][key]
        if key in {"reviewed", "dry_run"}:
            if value is True:
                args.append(flag)
            elif value is not False:
                raise LabError("确认选项必须为布尔值。")
        else:
            if isinstance(value, list):
                value = ",".join(map(str, value))
            args.append(f"{flag}={value}")
    return args


class Workspace:
    def __init__(self, root, python=None):
        self.root = Path(root).resolve()
        self.state_dir = self.root / ".gui"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.state_dir / "workspace.json"
        self.state = read_json(self.registry_file) if self.registry_file.is_file() else {"projects": [], "xanylabeling": ""}
        self.python = str(python or sys.executable)
        self.token = secrets.token_urlsafe(32)
        self.guard = threading.Lock()
        self.jobs = {}
        self.processes = {}
        self.started_at = time.time()
        for path in (self.state_dir / "jobs").glob("*/job.json"):
            job = read_json(path)
            self.jobs[job["id"]] = job
        # Discover user experiments only. Synthetic checks remain explicitly imported.
        for path in (self.root / "experiments").glob("*/project.json"):
            self.register(path.parent, save=False)
        self.save()
        with self.guard:
            self.reconcile_jobs()

    def reconcile_jobs(self):
        """Called under guard. Only adopt a PID with its original creation time."""
        for job in self.jobs.values():
            if job["status"] not in {"queued", "running", "cancelling", "recovering"}:
                continue
            folder = self.state_dir / "jobs" / job["id"]
            previous = dict(job)
            result_path, worker_path = folder / "result.json", folder / "worker.json"
            if result_path.is_file():
                result = read_json(result_path)
                if result.get("token") == job.get("worker_token"):
                    code = result["exit_code"]
                    job.update(status="completed" if code == 0 else "cancelled" if (folder / "cancel").exists() else "failed",
                               exit_code=code, completed_at=result["completed_at"], error=self.job_log(job["id"])[-3000:] if code else None)
                    self.processes.pop(job["id"], None)
                    write_json(folder / "job.json", job)
                    continue
            identity = job.get("worker")
            if worker_path.is_file():
                worker = read_json(worker_path)
                if worker.get("token") == job.get("worker_token"):
                    identity = worker
            # Old versions launched the CLI directly. Adopt only an exact argv match.
            if identity is None and not job.get("worker_token") and job.get("project_id"):
                expected = ["-m", "autolabel_lab", "--project", str(self.project(job["project_id"]).root),
                            *command_args(job["action"], job.get("payload", {}))]
                candidates = []
                created = datetime.fromisoformat(job.get("started_at", job["created_at"])).timestamp()
                for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
                    command = process.info["cmdline"] or []
                    if command[1:] == ["-u", *expected] and process.info["create_time"] is not None and process.info["create_time"] >= created - 2:
                        candidates.append(process)
                if len(candidates) == 1:
                    identity = {"pid": candidates[0].pid, "create_time": candidates[0].create_time()}
                    job["legacy_worker"] = True
            alive = False
            unknown = False
            if identity:
                job["worker"] = identity
                try:
                    process = psutil.Process(identity["pid"])
                    alive = abs(process.create_time() - identity["create_time"]) < .01 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied:
                    unknown = True
            elif job.get("worker_token"):
                # Allow the spawned worker to write its identity, including during restart.
                launched = datetime.fromisoformat(job["created_at"]).timestamp()
                unknown = time.time() - max(launched, self.started_at) < 30
                entry = next((p for p in self.state["projects"] if p["id"] == job["project_id"]), None)
                unknown = unknown or bool(entry and (Path(entry["path"]) / ".operation.lock").exists())
            else:
                entry = next((p for p in self.state["projects"] if p["id"] == job.get("project_id")), None)
                unknown = bool(entry and (Path(entry["path"]) / ".operation.lock").exists())
            if alive:
                job.update(status="cancelling" if (folder / "cancel").exists() else "running", error=None)
            elif unknown:
                job.update(status="cancelling" if (folder / "cancel").exists() else "recovering",
                           error="正在核验任务进程；确认结束前不会启动新任务。")
            else:
                # The worker may have exited just after writing result.json. Read it next poll.
                if result_path.is_file() and read_json(result_path).get("token") == job.get("worker_token"):
                    continue
                job.update(status="interrupted", completed_at=now(), error="任务进程已退出且没有完整结束记录；检查实验状态后使用新名称重试。")
                self.processes.pop(job["id"], None)
            if job != previous:
                write_json(folder / "job.json", job)

    def save(self):
        write_json(self.registry_file, self.state)

    def register(self, path, title=None, save=True):
        project = Project(path)
        with self.guard:
            for entry in self.state["projects"]:
                if entry["path"] == str(project.root):
                    return entry
            entry = {"id": uuid.uuid4().hex[:16], "title": title or project.root.name, "path": str(project.root)}
            self.state["projects"].append(entry)
            if save:
                self.save()
            return entry

    def project(self, project_id):
        entry = next((p for p in self.state["projects"] if p["id"] == project_id), None)
        if not entry:
            raise LabError("实验不存在。")
        return Project(entry["path"])

    def overview(self):
        projects = []
        for entry in self.state["projects"]:
            try:
                p = Project(entry["path"])
                summary = status(p)
                projects.append({**entry, "config": p.config, "summary": summary})
            except (LabError, OSError) as exc:
                projects.append({**entry, "error": str(exc)})
        packages = {}
        for package in ["ultralytics", "torch"]:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        with self.guard:
            self.reconcile_jobs()
            jobs = [dict(job) for job in self.jobs.values()]
        return {"app": "autolabel-lab", "workspace": str(self.root), "projects": projects, "jobs": jobs, "runtime": packages,
                "default_model": DEFAULT_MODEL_YAML, "xanylabeling": self.state.get("xanylabeling", "")}

    def details(self, project_id):
        p = self.project(project_id)
        models = []
        for path in sorted((p.root / "models").glob("*/model.json")):
            model = read_json(path)
            model["curve"] = curve(path.parent)
            evaluation = path.parent / "evaluation.json"
            model["evaluation"] = read_json(evaluation) if evaluation.is_file() else None
            models.append(model)
        models.sort(key=lambda model: model.get("created_at", ""))
        rounds = [read_json(path) for path in sorted((p.root / "rounds").glob("*/round.json"))]
        rounds.sort(key=lambda round_: round_.get("created_at", ""))
        counts = {}
        for role in ["seed", "val", "test"]:
            frames = p.frames(role)
            folder = p.root / "workspaces" / role
            counts[role] = {"total": len(frames), "saved": sum((folder / f"{f['id']}.json").is_file() for f in frames), "archived": role in p.annotations}
        report_path = p.root / "reports" / "report.json"
        result = read_json(report_path) if report_path.is_file() else None
        return {"config": p.config, "summary": status(p), "manifest": p.manifest, "models": models, "rounds": rounds,
                "annotations": p.annotations, "coverage": counts, "report": result,
                "timings": read_json(p.root / "timings.json"), "locked": (p.root / ".operation.lock").exists()}

    def start_job(self, project_id, action, payload):
        p = self.project(project_id)
        args = command_args(action, payload)
        with self.guard:
            self.reconcile_jobs()
            # The local machine has one GPU; avoid competing training processes.
            if any(j["status"] in {"queued", "running", "cancelling", "recovering"} for j in self.jobs.values()):
                raise LabError("已有任务正在执行，请等它完成后再启动下一项操作。")
            if (p.root / ".operation.lock").exists():
                raise LabError("实验已有其他任务在运行。异常中断留下的锁需确认无进程后手动移除。")
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "project_id": project_id, "action": action, "payload": payload,
                   "status": "queued", "created_at": now(), "error": None, "worker_token": secrets.token_urlsafe(24)}
            self.jobs[job_id] = job
            write_json(self.state_dir / "jobs" / job_id / "job.json", job)
            thread = threading.Thread(target=self.run_job, args=(p.root, job_id, args), daemon=True)
            thread.start()
        return job

    def run_job(self, project_root, job_id, args):
        job = self.jobs[job_id]
        folder = self.state_dir / "jobs" / job_id
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        env["ALAB_CANCEL_FILE"] = str(folder / "cancel")
        try:
            with (folder / "output.log").open("w", encoding="utf-8") as output:
                process = subprocess.Popen([self.python, "-u", "-m", "autolabel_lab.job_worker", str(folder), "--project", str(project_root), *args],
                                           stdout=output, stderr=subprocess.STDOUT, env=env, cwd=str(self.root),
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                with self.guard:
                    self.processes[job_id] = process
                    job.update(status="cancelling" if (folder / "cancel").exists() else "running", started_at=now())
                    try:
                        job["worker"] = {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time()}
                    except psutil.Error:
                        pass  # A very short task can finish before the server observes its PID.
                    write_json(folder / "job.json", job)
                code = process.wait()
            with self.guard:
                self.reconcile_jobs()
                if job["status"] in {"running", "queued", "recovering", "cancelling"}:
                    job.update(status="interrupted", exit_code=code, completed_at=now(), error="工作进程未写入完整结果，请检查日志。")
        except Exception as exc:
            with self.guard:
                job.update(status="failed", error=str(exc), completed_at=now())
        finally:
            with self.guard:
                self.processes.pop(job_id, None)
                write_json(folder / "job.json", job)

    def cancel_job(self, job_id):
        with self.guard:
            self.reconcile_jobs()
            job = self.jobs.get(job_id)
            if not job or job["status"] not in {"queued", "running", "cancelling", "recovering"}:
                raise LabError("此任务已结束。")
            if job["action"] not in {"train", "extract", "autolabel", "evaluate"}:
                raise LabError("此操作正在提交数据，请等待提交完成。")
            (self.state_dir / "jobs" / job_id / "cancel").touch()
            job["status"] = "cancelling"
            write_json(self.state_dir / "jobs" / job_id / "job.json", job)
            return job

    def job_log(self, job_id):
        if job_id not in self.jobs:
            raise LabError("任务不存在。")
        path = self.state_dir / "jobs" / job_id / "output.log"
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 40000))
            return ANSI.sub("", stream.read().decode("utf-8", errors="replace")).replace("\r", "\n")


class Handler(BaseHTTPRequestHandler):
    @property
    def workspace(self):
        return self.server.workspace

    def log_message(self, *_):
        pass

    def send(self, value, code=200, content_type="application/json; charset=utf-8"):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8") if not isinstance(value, bytes) else value
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def trusted(self):
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin not in {f"http://{h}" for h in allowed}):
            raise LabError("只接受本机工作台请求。")

    def do_GET(self):
        try:
            self.trusted()
            parsed = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if parsed.path == "/":
                return self.send((STATIC / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
            if parsed.path in {"/app.js", "/style.css"}:
                return self.send((STATIC / parsed.path[1:]).read_bytes(), content_type="text/javascript; charset=utf-8" if parsed.path.endswith(".js") else "text/css; charset=utf-8")
            if parsed.path == "/api/bootstrap":
                return self.send({"token": self.workspace.token, **self.workspace.overview()})
            if parsed.path == "/api/overview":
                return self.send(self.workspace.overview())
            if parsed.path == "/api/project":
                return self.send(self.workspace.details(query["id"]))
            if parsed.path == "/api/log":
                return self.send({"text": self.workspace.job_log(query["id"])})
            if parsed.path == "/api/file":
                p = self.workspace.project(query["id"])
                path = contained(p.root, query["path"])
                if path.suffix.lower() not in {".jpg", ".png", ".json", ".csv", ".md", ".txt"} or not path.is_file():
                    raise LabError("文件不可预览。")
                return self.send(path.read_bytes(), content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            return self.send({"error": "未找到资源。"}, 404)
        except (KeyError, LabError, OSError, ValueError) as exc:
            self.send({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            self.trusted()
            if self.headers.get("X-Lab-Token") != self.workspace.token:
                return self.send({"error": "页面已过期，请刷新。"}, 403)
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 1024 * 1024:
                raise LabError("请求过大。")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise LabError("请求必须为 JSON 对象。")
            path = urlparse(self.path).path
            ws = self.workspace
            if path == "/api/create":
                name = safe_name(body["name"])
                root = ws.root / "experiments" / name
                Project.create(root, body["classes"], seed=finite_number(body.get("seed", 42), "随机种子", True))
                return self.send(ws.register(root, title=body.get("title") or name))
            if path == "/api/create-yolo":
                from .image_import import dataset_classes
                if any(j["status"] in {"queued", "running", "cancelling", "recovering"} for j in ws.overview()["jobs"]):
                    raise LabError("已有任务正在执行，请等它完成后再导入。")
                counts = {key: finite_number(body.get(key, default), key, True)
                          for key, default in [("train_count", 200), ("val_count", 50), ("pool_count", 200), ("test_count", 50)]}
                if any(n < 1 for n in counts.values()):
                    raise LabError("各分组图片数必须为正整数。")
                name = safe_name(body["name"])
                root = ws.root / "experiments" / name
                Project.create(root, dataset_classes(body["source"]), seed=finite_number(body.get("seed", 42), "随机种子", True))
                entry = ws.register(root, title=body.get("title") or name)
                return self.send({**entry, "job": ws.start_job(entry["id"], "import-yolo", {"source": body["source"], **counts})})
            if path == "/api/register":
                return self.send(ws.register(body["path"]))
            if path == "/api/job":
                return self.send(ws.start_job(body["project_id"], body["action"], body.get("payload", {})))
            if path == "/api/cancel":
                return self.send(ws.cancel_job(body["id"]))
            if path == "/api/pick":
                if body["kind"] not in {"video", "directory", "app"}:
                    raise LabError("不支持的文件选择。")
                result = subprocess.run([ws.python, "-m", "autolabel_lab.dialog", body["kind"]], capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONIOENCODING": "utf-8"})
                if result.returncode:
                    raise LabError("无法打开系统文件选择器；可直接粘贴文件路径。")
                return self.send(json.loads(result.stdout))
            if path == "/api/preview-video":
                video = Path(body["path"]).resolve()
                if not video.is_file():
                    raise LabError("视频不存在。")
                capture = cv2.VideoCapture(str(video))
                try:
                    fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    if not capture.isOpened() or fps <= 0 or not np.isfinite(fps) or count <= 0:
                        raise LabError("无法读取视频。")
                    second = finite_number(body.get("second", 0), "预览时间")
                    if not 0 <= second < count / fps:
                        raise LabError("预览时间超出视频范围。")
                    capture.set(cv2.CAP_PROP_POS_FRAMES, min(int(second * fps), int(count) - 1))
                    ok, pixels = capture.read()
                    if not ok:
                        raise LabError("无法解码该帧。")
                    height, width = pixels.shape[:2]
                    if width > 1000:
                        pixels = cv2.resize(pixels, (1000, round(height * 1000 / width)))
                    import base64
                    encoded = cv2.imencode(".jpg", pixels)[1].tobytes()
                    return self.send({"duration": count / fps, "fps": fps, "width": width, "height": height,
                                      "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()})
                finally:
                    capture.release()
            if path == "/api/settings":
                app = Path(body["xanylabeling"]).resolve()
                if not app.is_file() or app.suffix.lower() != ".exe":
                    raise LabError("请选择 X-AnyLabeling 的 .exe 程序。")
                with ws.guard:
                    ws.state["xanylabeling"] = str(app)
                    ws.save()
                return self.send({"path": str(app)})
            if path == "/api/open-folder":
                p = ws.project(body["project_id"])
                location = body.get("location", "root")
                mapping = {"root": "", "seed": "workspaces/seed", "val": "workspaces/val", "test": "workspaces/test", "reports": "reports"}
                if location == "round":
                    relative = f"rounds/{safe_name(body['round'])}/editable"
                elif location in mapping:
                    relative = mapping[location]
                else:
                    raise LabError("未知目录。")
                folder = contained(p.root, relative)
                if not folder.is_dir():
                    raise LabError("目录尚未生成。")
                if os.name == "nt":
                    os.startfile(str(folder))
                else:
                    subprocess.Popen(["xdg-open", str(folder)])
                if body.get("launch_app"):
                    executable = ws.state.get("xanylabeling", "")
                    if not executable or not Path(executable).is_file():
                        raise LabError("请先在右上角设置 X-AnyLabeling 程序。目录已打开。")
                    subprocess.Popen([executable], creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                return self.send({"path": str(folder)})
            return self.send({"error": "未知操作。"}, 404)
        except (KeyError, LabError, OSError, ValueError, TypeError) as exc:
            self.send({"error": str(exc)}, 400)


def make_server(root, port=8765, python=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.workspace = Workspace(root, python=python)
    return server


def main():
    parser = argparse.ArgumentParser(description="自动标注实验可视化工作台")
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = make_server(args.workspace, args.port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"实验工作台：{url}\n关闭此进程会关闭网页服务；训练任务请在网页中停止后再退出。", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

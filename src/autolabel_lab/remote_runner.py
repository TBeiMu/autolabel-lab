"""Server-side detached trainer and rank entry point, copied with each job."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def probe():
    import importlib.metadata
    import torch
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    gpus = []
    for row in rows.strip().splitlines():
        index, name, total, free, usage = [v.strip() for v in row.split(",")]
        gpus.append({"index": int(index), "name": name, "total_mib": int(total), "free_mib": int(free), "utilization": int(usage)})
    return {"python": sys.executable, "python_version": sys.version.split()[0],
            "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "ultralytics": importlib.metadata.version("ultralytics"), "gpus": gpus}


def cancelled(root):
    if (Path(root) / "cancel").exists():
        raise RuntimeError("用户请求停止远程训练。")


def rank_train(root):
    from ultralytics.models.yolo.obb.train import OBBTrainer
    from ultralytics.utils.torch_utils import init_seeds
    root = Path(root)
    request = json.loads((root / "request.json").read_text())
    # This trainer runs in every DDP rank, preserving scratch initialization and cancellation.
    class LabTrainer(OBBTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            init_seeds(request["params"]["seed"], deterministic=True)
            return super().get_model(cfg, weights if request["init"] == "continue" else None, verbose)
    overrides = dict(request["params"], model=request["remote_model"], data=str(root / "dataset/data.yaml"),
                     project=str(root / "runs"), name="train", exist_ok=True)
    trainer = LabTrainer(overrides=overrides)
    for event in ["on_train_batch_start", "on_train_epoch_start", "on_val_batch_start"]:
        trainer.add_callback(event, lambda *args: cancelled(root))
    try:
        trainer.train()
    finally:
        import torch.distributed as distributed
        if distributed.is_initialized():
            distributed.destroy_process_group()


def supervise(root):
    root = Path(root).resolve()
    request = json.loads((root / "request.json").read_text())
    locks = []
    started = time.monotonic()
    result = {"token": request["token"], "status": "failed"}
    child = None
    try:
        cancelled(root)
        inventory = probe()
        if inventory["ultralytics"] != "8.4.91" or not inventory["cuda_available"]:
            raise RuntimeError("远程环境必须使用 Ultralytics 8.4.91 和可用 CUDA。")
        result["runtime"] = inventory
        for index in request["params"]["device"].split(","):
            if int(index) not in {g["index"] for g in inventory["gpus"]}:
                raise RuntimeError("所选 GPU 不存在。")
            lock_dir = Path.home() / ".cache/autolabel-lab/gpu-locks"
            lock_dir.mkdir(parents=True, exist_ok=True)
            stream = (lock_dir / f"gpu-{index}.lock").open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                raise RuntimeError(f"GPU {index} 正在执行另一个工作台训练。")
            locks.append(stream)
        for name, expected in request["transfer_hashes"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or digest(path) != expected:
                raise RuntimeError(f"传输文件校验失败：{name}")
        import yaml
        data = {"path": str(root / "dataset"), "train": "images/train", "val": "images/val",
                "names": dict(enumerate(request["classes"]))}
        (root / "dataset/data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True))
        count = len(request["params"]["device"].split(","))
        rank_cmd = ["-m", "autolabel_lab.remote_runner", "rank", str(root)]
        if count > 1:
            rank_cmd = ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={count}", *rank_cmd]
        (root / "yolo-config").mkdir(exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(root / "code"), "PYTHONUNBUFFERED": "1",
               "YOLO_CONFIG_DIR": str(root / "yolo-config"), "NCCL_P2P_DISABLE": "1"}
        child = subprocess.Popen([sys.executable, *rank_cmd], cwd=root, env=env, start_new_session=True)
        write(root / "child.json", {"pid": child.pid, "start": process_start(child.pid), "token": request["token"]})
        cancel_at = None
        while child.poll() is None:
            if (root / "cancel").exists():
                cancel_at = cancel_at or time.monotonic()
                if time.monotonic() - cancel_at > 15:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
            time.sleep(.5)
        if (root / "cancel").exists():
            result.update(status="cancelled", error="远程训练已停止。")
        elif child.returncode:
            result.update(error=f"远程训练进程退出码：{child.returncode}，请查看日志。")
        else:
            weights = root / "runs/train/weights/best.pt"
            if not weights.is_file():
                raise RuntimeError("远程训练没有生成 best.pt。")
            result.update(status="completed", weights_sha256=digest(weights))
    except BaseException as error:
        import traceback
        traceback.print_exc()
        result.update(error=str(error), status="cancelled" if (root / "cancel").exists() else "failed")
    finally:
        if child and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        result["machine_seconds"] = time.monotonic() - started
        write(root / "result.json", result)
        for stream in locks:
            stream.close()


def process_start(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def launch(root):
    root = Path(root).resolve()
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "supervisor.json").exists():
            return json.loads((root / "supervisor.json").read_text())
        request = json.loads((root / "request.json").read_text())
        with (root / "output.log").open("ab") as output:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "supervise", str(root)],
                                       cwd=root, stdout=output, stderr=output, start_new_session=True)
        identity = {"pid": process.pid, "start": process_start(process.pid), "token": request["token"]}
        write(root / "supervisor.json", identity)
        return identity


def status(root, offset=0):
    root = Path(root).resolve()
    identity_path = root / "supervisor.json"
    result_path = root / "result.json"
    alive = False
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        try:
            alive = process_start(identity["pid"]) == identity["start"]
        except OSError:
            pass
    log = root / "output.log"
    text, size = "", offset
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(offset)
            text = stream.read(100000).decode("utf-8", errors="replace")
            size = stream.tell()
    csv_path = root / "runs/train/results.csv"
    return {"alive": alive, "result": json.loads(result_path.read_text()) if result_path.exists() else None,
            "log": text, "offset": size, "csv": csv_path.read_text() if csv_path.exists() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["probe", "launch", "supervise", "rank", "status", "cancel"])
    parser.add_argument("root", nargs="?")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.action == "probe":
        print(json.dumps(probe()))
    elif args.action == "supervise":
        supervise(args.root)
    elif args.action == "rank":
        rank_train(args.root)
    elif args.action == "launch":
        print(json.dumps(launch(args.root)))
    elif args.action == "cancel":
        (Path(args.root) / "cancel").touch()
        print(json.dumps({"cancel_requested": True}))
    else:
        print(json.dumps(status(args.root, args.offset)))


if __name__ == "__main__":
    main()

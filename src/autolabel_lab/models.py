from __future__ import annotations

import importlib.metadata
import math
import platform
import shutil
import time
from pathlib import Path

from .dataset import build_dataset, sample_evenly
from .cancellation import check_cancelled
from .labels import make_document, quad
from .project import LabError, now, safe_name, sha256, write_json


def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise LabError("当前 Python 没有训练依赖。运行 scripts/setup.ps1 -Training，或设置 ALAB_PYTHON 为已安装 Ultralytics 的 Python。") from exc
    return YOLO


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ["ultralytics", "torch", "opencv-python", "opencv-python-headless", "numpy"]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def train(project, name, rounds=(), init="scratch", from_model=None, model_yaml=None,
          epochs=100, imgsz=640, batch=4, device="cpu", dry_run=False):
    name = safe_name(name)
    model_yaml = model_yaml or project.config["model_yaml"]
    if init == "scratch" and Path(model_yaml).suffix.lower() not in {".yaml", ".yml"}:
        raise LabError("从零训练只接受模型结构 YAML，不能提供 .pt 权重。")
    if epochs < 1 or imgsz < 32 or imgsz % 32 or batch < 1:
        raise LabError("epochs、batch 必须为正数，imgsz 必须是至少 32 的 32 倍数。")
    if init == "continue" and not from_model:
        raise LabError("继续训练需要 --from-model M0。")
    if init == "scratch" and from_model:
        raise LabError("从零训练不能指定父模型；需要继续训练时选择 --init continue。")
    root = project.root / "models" / name
    if root.exists():
        raise LabError(f"模型名称已使用：{name}，请换新名称。")
    if from_model:
        parent, parent_weights = project.model(from_model)
        if model_yaml != parent["model_yaml"] and model_yaml != project.config["model_yaml"]:
            raise LabError("继续训练不能更换模型结构。")
        model_yaml = parent["model_yaml"]
        # A continued model cannot drop labels that its parent has already seen.
        if not set(parent["rounds"]).issubset(set(rounds)):
            raise LabError("继续训练必须保留父模型已经使用的所有审核轮次。")
    if not dry_run:
        YOLO = load_yolo()
    dataset, snapshot = build_dataset(project, name, list(rounds))
    params = {"epochs": epochs, "imgsz": imgsz, "batch": batch, "device": device,
              "seed": project.config["seed"], "workers": 0, "pretrained": init == "continue", "resume": False,
              "amp": False, "deterministic": True, "optimizer": "AdamW", "lr0": 0.001,
              "patience": 0, "close_mosaic": 0, "plots": False}
    record = {"name": name, "init": init, "from_model": from_model, "model_yaml": model_yaml,
              "rounds": list(rounds), "params": params, "dataset": dataset.relative_to(project.root).as_posix(),
              "dataset_hash": snapshot["dataset_hash"], "test_hash": snapshot["test_hash"],
              "counts": snapshot["counts"], "status": "planned", "created_at": now(), "versions": versions()}
    if from_model:
        record["parent_weights_sha256"] = parent["weights_sha256"]
    if dry_run:
        write_json(project.root / "reports" / f"{name}_plan.json", record)
        return record
    root.mkdir(parents=True)
    record["status"] = "running"
    write_json(root / "model.json", record)
    started = time.perf_counter()
    try:
        from ultralytics.utils.torch_utils import init_seeds
        # Seed before building the network, not just inside Trainer.
        init_seeds(project.config["seed"], deterministic=True)
        model = YOLO(model_yaml if init == "scratch" else str(parent_weights), task="obb")
        if model.task != "obb":
            raise LabError("模型不是 OBB 旋转框模型。")
        for event in ["on_train_batch_start", "on_train_epoch_start", "on_val_batch_start"]:
            model.add_callback(event, check_cancelled)
        check_cancelled()
        model.train(data=str(dataset / "data.yaml"), project=str(root / "runs"), name="train", exist_ok=False, **params)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise LabError("训练结束但未产生 best.pt，请检查训练日志。")
        output = root / "best.pt"
        shutil.copy2(best, output)
        record.update(status="completed", completed_at=now(), weights=output.relative_to(project.root).as_posix(),
                      weights_sha256=sha256(output), machine_seconds=time.perf_counter() - started)
        write_json(root / "model.json", record)
        return record
    except BaseException as exc:
        record.update(status="failed", error=str(exc), machine_seconds=time.perf_counter() - started)
        write_json(root / "model.json", record)
        raise


def autolabel(project, name, model_name, count=25, frame_ids=None, conf=0.25, iou=0.7, imgsz=None, device="cpu"):
    project.require_split()
    name = safe_name(name)
    if not (math.isfinite(conf) and 0 < conf <= 1 and math.isfinite(iou) and 0 < iou <= 1):
        raise LabError("置信度和 NMS IoU 必须在 (0,1] 内。")
    if imgsz is not None and (imgsz < 32 or imgsz % 32):
        raise LabError("imgsz 必须是至少 32 的 32 倍数。")
    root = project.root / "rounds" / name
    if root.exists():
        raise LabError("轮次名称已使用，请使用新名称。")
    model_record, weights = project.model(model_name)
    from .project import read_json
    used = set()
    for path in (project.root / "rounds").glob("*/round.json"):
        existing = read_json(path)
        if existing["status"] in {"running", "predicted", "reviewed"}:
            used.update(existing["frame_ids"])
    pool = sorted([f for f in project.frames("pool") if f["id"] not in used], key=lambda f: (f["source_sha256"], f["timestamp"]))
    if frame_ids:
        if len(set(frame_ids)) != len(frame_ids) or not set(frame_ids).issubset({f["id"] for f in pool}):
            raise LabError("指定图片必须是未分配的训练池图片；不能自动标注验证/测试集。")
        selected = [f for f in pool if f["id"] in frame_ids]
    else:
        selected = sample_evenly(pool, count)
    YOLO = load_yolo()
    model = YOLO(str(weights), task="obb")
    if model.task != "obb" or [model.names[i] for i in range(len(model.names))] != project.config["classes"]:
        raise LabError("模型类别顺序或任务与实验不一致。")
    for frame in selected:
        project.image_path(frame)
    (root / "raw").mkdir(parents=True)
    (root / "editable").mkdir()
    params = {"conf": conf, "iou": iou, "imgsz": imgsz or model_record["params"]["imgsz"], "device": device}
    record = {"name": name, "model": model_name, "model_weights_sha256": model_record["weights_sha256"],
              "frame_ids": [f["id"] for f in selected], "params": params, "status": "running",
              "created_at": now(), "raw_hashes": {}}
    write_json(root / "round.json", record)
    started = time.perf_counter()
    try:
        for frame in selected:
            check_cancelled()
            result = model.predict(source=str(project.image_path(frame)), verbose=False, save=False, **params)[0]
            if result.obb is None:
                raise LabError("模型未返回 OBB 结果。")
            points = result.obb.xyxyxyxy.cpu().numpy()
            classes = result.obb.cls.cpu().numpy().astype(int)
            scores = result.obb.conf.cpu().numpy()
            shapes = [{"label": project.config["classes"][int(c)], "points": quad(p, frame["width"], frame["height"], check_bounds=False),
                       "score": float(s)} for p, c, s in zip(points, classes, scores)]
            document = make_document(frame, shapes, prediction=True)
            raw = root / "raw" / f"{frame['id']}.json"
            write_json(raw, document)
            record["raw_hashes"][frame["id"]] = sha256(raw)
            shutil.copy2(raw, root / "editable" / raw.name)
            shutil.copy2(project.image_path(frame), root / "editable" / frame["file"])
        (root / "editable" / "classes.txt").write_text("\n".join(project.config["classes"]) + "\n", encoding="utf-8")
        record.update(status="predicted", completed_at=now(), machine_seconds=time.perf_counter() - started)
        write_json(root / "round.json", record)
        return {**record, "open_in_x_anylabeling": str(root / "editable")}
    except BaseException as exc:
        record.update(status="failed", error=str(exc), machine_seconds=time.perf_counter() - started)
        write_json(root / "round.json", record)
        raise


def evaluate(project, names, device="cpu"):
    names = list(dict.fromkeys(safe_name(n) for n in names))
    if not names:
        raise LabError("请指定至少一个模型。")
    plans = []
    for name in names:
        record, weights = project.model(name)
        dataset, snapshot = build_dataset(project, name, record["rounds"])
        if snapshot["dataset_hash"] != record["dataset_hash"] or snapshot["test_hash"] != record["test_hash"]:
            raise LabError("模型的数据快照与当前固定数据不一致。")
        if (project.root / "models" / name / "evaluation.json").exists():
            raise LabError(f"{name} 已评价；report 会显示已保存的结果。")
        plans.append((name, record, weights, dataset))
    if len({record["test_hash"] for _, record, _, _ in plans}) != 1:
        raise LabError("模型测试集不一致，拒绝比较。")
    if len({record["params"]["imgsz"] for _, record, _, _ in plans}) != 1:
        raise LabError("比较模型的输入尺寸不同；请使用相同训练设置。")
    YOLO = load_yolo()
    results = []
    for name, record, weights, dataset in plans:
        check_cancelled()
        model = YOLO(str(weights), task="obb")
        model.add_callback("on_val_batch_start", check_cancelled)
        started = time.perf_counter()
        params = {"split": "test", "imgsz": record["params"]["imgsz"], "device": device,
                  "conf": 0.001, "iou": 0.7, "workers": 0, "plots": False, "save_json": False}
        metrics = model.val(data=str(dataset / "data.yaml"), project=str(project.root / "models" / name / "test_run"),
                            name="test", exist_ok=False, **params)
        values = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                  "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)}
        if not all(math.isfinite(v) for v in values.values()):
            raise LabError("评价返回非有限指标，请检查测试标注。")
        result = {"model": name, "test_hash": record["test_hash"], "params": params,
                  "metrics": values, "created_at": now(), "machine_seconds": time.perf_counter() - started,
                  "weights_sha256": record["weights_sha256"]}
        write_json(project.root / "models" / name / "evaluation.json", result)
        results.append(result)
    return results

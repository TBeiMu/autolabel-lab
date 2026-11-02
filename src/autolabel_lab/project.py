from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_YAML = "yolo26s-obb.yaml"


class LabError(ValueError):
    """An actionable experiment error."""


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise LabError(f"无法读取 JSON：{path}：{exc}") from exc


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) or value.upper().split(".")[0] in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))
    }:
        raise LabError("名称仅允许英文字母、数字、短横线、下划线，且不能是 Windows 保留名。")
    return value


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = read_json(self.root / "project.json")
        if self.config.get("schema_version") != 1 or self.config.get("task") != "obb":
            raise LabError("不支持的实验配置。")

    @classmethod
    def create(cls, root, classes, seed=42):
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise LabError(f"实验目录必须为空：{root}")
        if not classes or any(not c.strip() or c != c.strip() or any(x in c for x in "\r\n\t") for c in classes) or len(set(classes)) != len(classes):
            raise LabError("类别必须非空、无重复，且不含换行或首尾空格。")
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise LabError("随机种子必须是 0 到 2^32-1 之间的整数。")
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "project.json", {"schema_version": 1, "task": "obb", "classes": classes,
                   "seed": seed, "model_yaml": DEFAULT_MODEL_YAML, "created_at": now()})
        write_json(root / "manifest.json", {"frames": [], "split": None})
        write_json(root / "annotations.json", {})
        write_json(root / "timings.json", [])
        for directory in ["frames", "workspaces", "annotations", "rounds", "models", "reports", "datasets"]:
            (root / directory).mkdir(exist_ok=True)
        return cls(root)

    @property
    def manifest(self):
        return read_json(self.root / "manifest.json")

    @property
    def annotations(self):
        return read_json(self.root / "annotations.json")

    def frames(self, subset=None):
        return [f for f in self.manifest["frames"] if subset is None or f["role"] == subset]

    def require_split(self):
        manifest = self.manifest
        if not manifest["split"]:
            raise LabError("请先执行 split 固定数据划分。")
        if fingerprint(manifest["frames"]) != manifest["split"]["frames_hash"]:
            raise LabError("数据清单在划分后被修改；请新建实验，不能改变固定划分。")

    def require_annotations(self, subset):
        info = self.annotations.get(subset)
        if not info:
            raise LabError(f"{subset} 尚未完成人工标注导入。")
        folder = self.root / info["folder"]
        for frame_id, digest in info["hashes"].items():
            path = folder / f"{frame_id}.json"
            if not path.is_file() or sha256(path) != digest:
                raise LabError(f"已归档标注被修改：{path}")
        return info

    def image_path(self, frame):
        path = self.root / "frames" / frame["file"]
        if not path.is_file() or sha256(path) != frame["image_sha256"]:
            raise LabError(f"原始图片缺失或被修改：{path}")
        return path

    def round(self, name):
        return read_json(self.root / "rounds" / safe_name(name) / "round.json")

    def model(self, name):
        record = read_json(self.root / "models" / safe_name(name) / "model.json")
        if record["status"] != "completed":
            raise LabError(f"模型 {name} 未训练完成。")
        path = self.root / record["weights"]
        if not path.is_file() or sha256(path) != record["weights_sha256"]:
            raise LabError(f"模型权重缺失或被修改：{path}")
        return record, path

    @contextmanager
    def lock(self):
        path = self.root / ".operation.lock"
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(now())
        except FileExistsError as exc:
            raise LabError("另一个操作正在运行。如果此前进程异常退出，确认无任务后删除 .operation.lock。") from exc
        try:
            yield
        finally:
            path.unlink(missing_ok=True)

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from autolabel_lab.dataset import build_dataset
from autolabel_lab.image_import import dataset_classes, import_yolo
from autolabel_lab.models import autolabel
from autolabel_lab.project import LabError, Project, read_json, sha256, write_json
from autolabel_lab.reports import report


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"path": "/old/linux/path", "nc": 1, "names": {0: "target"}}))
    for role, count in [("train", 12), ("val", 4)]:
        (root / "images" / role).mkdir(parents=True)
        (root / "labels" / role).mkdir(parents=True)
        for i in range(count):
            pixels = np.full((32, 32, 3), i + (100 if role == "val" else 0), np.uint8)
            image = root / "images" / role / f"frame_{i:04d}.png"
            image.write_bytes(cv2.imencode(".png", pixels)[1].tobytes())
            (root / "labels" / role / f"frame_{i:04d}.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
    return root


def prepare(tmp_path, source, name="experiment"):
    project = Project.create(tmp_path / name, dataset_classes(source))
    result = import_yolo(project, source, train_count=4, val_count=2, pool_count=4, test_count=2)
    return project, result


def test_fixed_selection_preserves_sources_and_excludes_reference_labels(tmp_path, source):
    original = {str(p): sha256(p) for p in source.rglob("*") if p.is_file()}
    project, result = prepare(tmp_path, source)
    assert result["unused_images"] == 4
    assert {str(p): sha256(p) for p in source.rglob("*") if p.is_file()} == original
    project.require_split()
    assert len({f["id"] for f in project.frames()}) == 12
    assert all(f["timestamp"] is None for f in project.frames())
    assert set(project.annotations) == {"seed", "val", "test"}
    assert all("/val/" in f["source"].replace("\\", "/") for f in project.frames("val"))
    assert len(list((project.root / "references/pool").glob("*.json"))) == 4
    assert not list((project.root / "rounds").glob("*/editable/*.json"))
    dataset, snapshot = build_dataset(project, "M0", [])
    assert snapshot["counts"] == {"train": 4, "val": 2, "test": 2}
    training = {p.stem for p in (dataset / "labels/train").glob("*.txt")}
    assert training == {f["id"] for f in project.frames("seed")}
    assert not training.intersection(f["id"] for f in project.frames("pool"))
    other, _ = prepare(tmp_path, source, "repeat")
    assert [(f["id"], f["role"]) for f in other.frames()] == [(f["id"], f["role"]) for f in project.frames()]
    report(project)
    assert "同一动作" in (project.root / "reports/report.md").read_text(encoding="utf-8")


def test_malformed_or_missing_labels_prevent_fixed_import(tmp_path, source):
    missing = source / "labels/train/frame_0001.txt"
    missing.unlink()
    project = Project.create(tmp_path / "invalid", ["target"])
    with pytest.raises(LabError, match="缺少标注"):
        import_yolo(project, source, 4, 2, 4, 2)
    assert project.manifest["split"] is None
    assert project.annotations == {}
    missing.write_text("0 0.1 0.2 0.3 0.4\n")
    with pytest.raises(LabError, match="8 个"):
        import_yolo(project, source, 4, 2, 4, 2)


def test_identical_pixels_cannot_cross_subsets(tmp_path, source):
    (source / "images/val/frame_0000.png").write_bytes((source / "images/train/frame_0000.png").read_bytes())
    project, _ = prepare(tmp_path, source)
    assert len({f["pixel_sha256"] for f in project.frames()}) == len(project.frames())
    assert read_json(project.root / "source_import.json")["duplicates_skipped"] == ["images/val/frame_0000.png"]


def test_prediction_workspace_contains_model_output_without_reference_truth(tmp_path, source, monkeypatch):
    project, _ = prepare(tmp_path, source)
    weights = project.root / "models/M0/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"test double, not a trained model")
    write_json(weights.parent / "model.json", {"status": "completed", "weights": "models/M0/best.pt",
               "weights_sha256": sha256(weights), "params": {"imgsz": 32}})
    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def cpu(self): return self
        def numpy(self): return self.data
    class FakeModel:
        task = "obb"
        names = {0: "target"}
        def predict(self, **kwargs):
            return [SimpleNamespace(obb=SimpleNamespace(xyxyxyxy=Tensor(np.empty((0, 4, 2))),
                                     cls=Tensor([]), conf=Tensor([])))]
    monkeypatch.setattr("autolabel_lab.models.load_yolo", lambda: lambda *a, **k: FakeModel())
    record = autolabel(project, "r1", "M0", count=4)
    for fid in record["frame_ids"]:
        assert read_json(project.root / f"rounds/r1/editable/{fid}.json")["shapes"] == []
        assert len(read_json(project.root / f"references/pool/{fid}.json")["shapes"]) == 1

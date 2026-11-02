import copy
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from autolabel_lab.cli import main
from autolabel_lab.dataset import build_dataset, extract, import_labels, split
from autolabel_lab.labels import compare_shapes, make_document, parse_document, polygon_iou, yolo_text
from autolabel_lab.models import autolabel, train
from autolabel_lab.project import LabError, Project, read_json, sha256, write_json
from autolabel_lab.reports import record_time, report


@pytest.fixture
def project(tmp_path):
    video = tmp_path / "输入 视频.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (96, 96))
    assert writer.isOpened()
    for i in range(100):
        image = np.full((96, 96, 3), 220, np.uint8)
        cv2.rectangle(image, (10 + i % 40, 20), (30 + i % 40, 40), (0, 0, 180), -1)
        cv2.putText(image, str(i), (4, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        writer.write(image)
    writer.release()
    p = Project.create(tmp_path / "实验", ["目标"])
    extract(p, video, every=0.4)
    split(p, seed_count=4, gap_seconds=0.4)
    return p


def save_manual(project, subset, empty=False):
    for frame in project.frames(subset):
        shapes = [] if empty else [{"label": "目标", "points": [[10, 20], [30, 20], [30, 40], [10, 40]]}]
        write_json(project.root / "workspaces" / subset / f"{frame['id']}.json", make_document(frame, shapes))


def import_all(project):
    for subset in ["seed", "val", "test"]:
        save_manual(project, subset)
        import_labels(project, subset=subset)


def test_temporal_split_has_gap_and_disjoint_roles(project):
    assert len(project.frames("seed")) == 4
    source = project.frames()[0]["source_sha256"]
    train_times = [f["timestamp"] for f in project.frames() if f["source_sha256"] == source and f["role"] in {"seed", "pool"}]
    val_times = [f["timestamp"] for f in project.frames("val")]
    test_times = [f["timestamp"] for f in project.frames("test")]
    assert min(val_times) - max(train_times) >= 0.4 - 1e-6
    assert min(test_times) - max(val_times) >= 0.4 - 1e-6
    with pytest.raises(LabError, match="已固定"):
        split(project)


def test_missing_annotation_is_not_negative_and_import_is_atomic(project):
    save_manual(project, "seed")
    missing = project.root / "workspaces" / "seed" / f"{project.frames('seed')[-1]['id']}.json"
    missing.unlink()
    with pytest.raises(LabError, match="缺少人工标注"):
        import_labels(project, subset="seed")
    assert "seed" not in project.annotations
    assert not (project.root / "annotations" / "seed").exists()


def test_explicit_empty_annotations_export_empty_yolo(project):
    save_manual(project, "val", empty=True)
    result = import_labels(project, subset="val")
    assert result["boxes"] == 0
    assert len(project.require_annotations("val")["hashes"]) == len(project.frames("val"))
    assert yolo_text(project.frames("val")[0], [], ["目标"]) == ""


def test_dimensions_class_and_bounds_are_checked(project):
    frame = project.frames("seed")[0]
    shapes = [{"label": "目标", "points": [[10, 10], [20, 10], [20, 20], [10, 20]]}]
    doc = make_document(frame, shapes)
    bad = copy.deepcopy(doc)
    bad["imageWidth"] += 1
    with pytest.raises(LabError, match="尺寸"):
        parse_document(bad, frame, ["目标"])
    bad = copy.deepcopy(doc)
    bad["shapes"][0]["label"] = "拼写错误"
    with pytest.raises(LabError, match="未知类别"):
        parse_document(bad, frame, ["目标"])
    bad = copy.deepcopy(doc)
    bad["shapes"][0]["points"][0][0] = -1
    with pytest.raises(LabError, match="边界"):
        parse_document(bad, frame, ["目标"])
    bad = copy.deepcopy(doc)
    bad["shapes"][0]["points"][0][0] = math.nan
    with pytest.raises(LabError, match="有限"):
        parse_document(bad, frame, ["目标"])


def test_rectangle_conversion_and_corner_normalization(project):
    frame = project.frames("seed")[0]
    doc = make_document(frame, [{"label": "目标", "points": [[20, 20], [10, 20], [10, 10], [20, 10]]}])
    doc["shapes"][0].update(shape_type="rectangle", points=[[10, 10], [20, 20]])
    shapes = parse_document(doc, frame, ["目标"])
    assert polygon_iou(shapes[0]["points"], [[10, 10], [20, 10], [20, 20], [10, 20]]) == pytest.approx(1)
    text = yolo_text(frame, shapes, ["目标"])
    assert len(text.split()) == 9
    assert all(0 <= float(v) <= 1 for v in text.split()[1:])


def test_archive_and_manifest_tampering_are_detected(project):
    import_all(project)
    info = project.annotations["seed"]
    path = project.root / info["folder"] / f"{info['frame_ids'][0]}.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(LabError, match="被修改"):
        build_dataset(project, "M0", [])
    manifest = project.manifest
    manifest["frames"][0]["role"] = "pool"
    write_json(project.root / "manifest.json", manifest)
    with pytest.raises(LabError, match="清单"):
        project.require_split()


def test_snapshot_integrity_and_fixed_test_hash(project):
    import_all(project)
    folder, a = build_dataset(project, "A", [])
    _, b = build_dataset(project, "B", [])
    assert a["test_hash"] == b["test_hash"]
    assert a["counts"]["train"] == 4
    label = next((folder / "labels" / "train").glob("*.txt"))
    label.write_text("", encoding="utf-8")
    with pytest.raises(LabError, match="快照被修改"):
        build_dataset(project, "A", [])


def test_dry_run_has_no_model_and_rejects_pretrained_weights(project):
    import_all(project)
    result = train(project, "M0", epochs=1, imgsz=64, batch=2, dry_run=True)
    assert result["init"] == "scratch"
    assert result["model_yaml"] == "yolo26s-obb.yaml"
    assert Project(project.root).config["model_yaml"] == "yolo26s-obb.yaml"
    assert result["params"]["pretrained"] is False
    assert result["params"]["amp"] is False
    assert result["status"] == "planned"
    assert not (project.root / "models" / "M0").exists()
    with pytest.raises(LabError, match="结构 YAML"):
        train(project, "bad", model_yaml="yolo26s-obb.pt", dry_run=True)


def test_obb_matching_counts_each_box_once():
    box = lambda x, label="目标": {"label": label, "points": [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]]}
    result = compare_shapes([box(0), box(0), box(30, "错类"), box(80)], [box(0), box(30), box(50)])
    assert result["unchanged"] == 1
    assert result["class_changed"] == 1
    assert result["false_positive"] == 2
    assert result["missed"] == 1


def make_review_round(project, name="r1"):
    selected = project.frames("pool")[:2]
    raw_hashes = {}
    from autolabel_lab.project import sha256
    for frame in selected:
        shapes = [{"label": "目标", "points": [[10, 20], [30, 20], [30, 40], [10, 40]], "score": 0.7}]
        doc = make_document(frame, shapes, prediction=True)
        raw = project.root / "rounds" / name / "raw" / f"{frame['id']}.json"
        write_json(raw, doc)
        raw_hashes[frame["id"]] = sha256(raw)
        write_json(project.root / "rounds" / name / "editable" / raw.name, doc)
    write_json(project.root / "rounds" / name / "round.json", {"name": name, "model": "M0", "status": "predicted",
               "frame_ids": [f["id"] for f in selected], "raw_hashes": raw_hashes, "params": {"conf": 0.25}})
    return selected


def test_review_requires_human_confirmation_and_preserves_raw(project):
    frames = make_review_round(project)
    with pytest.raises(LabError, match="逐张审核"):
        import_labels(project, round_name="r1")
    import_labels(project, round_name="r1", reviewed=True)
    assert project.round("r1")["status"] == "reviewed"
    result = report(project)
    assert result["reviewed_rounds"] == 1
    data = read_json(project.root / "reports" / "report.json")
    assert data["rounds"][0]["counts"]["unchanged"] == len(frames)
    assert data["rounds"][0]["human_time"]["reported_time_reduction"] is None
    raw = project.root / "rounds" / "r1" / "raw" / f"{frames[0]['id']}.json"
    raw.write_text("{}", encoding="utf-8")
    with pytest.raises(LabError, match="原始预测"):
        report(project)


def test_timing_requires_full_coverage_before_savings(project):
    frames = make_review_round(project)
    import_labels(project, round_name="r1", reviewed=True)
    record_time(project, "manual", 100, round_name="r1")
    record_time(project, "review", 10, round_name="r1", frame_id=frames[0]["id"])
    report(project)
    r = read_json(project.root / "reports" / "report.json")["rounds"][0]
    assert r["human_time"]["reported_time_reduction"] is None
    record_time(project, "review", 10, round_name="r1", frame_id=frames[1]["id"])
    report(project)
    r = read_json(project.root / "reports" / "report.json")["rounds"][0]
    assert r["human_time"]["reported_time_reduction"] == pytest.approx(0.8)


def test_reviewed_rounds_expand_train_but_keep_test(project):
    import_all(project)
    _, first = build_dataset(project, "M0", [])
    make_review_round(project)
    import_labels(project, round_name="r1", reviewed=True)
    _, second = build_dataset(project, "M1", ["r1"])
    assert second["counts"]["train"] == first["counts"]["train"] + 2
    assert first["test_hash"] == second["test_hash"]
    assert first["dataset_hash"] != second["dataset_hash"]


def test_cli_rejects_malformed_inputs_and_removes_lock(project, capsys):
    with pytest.raises(SystemExit) as result:
        main(["--project", str(project.root), "time", "--stage", "seed", "--seconds", "nan"])
    assert result.value.code == 2
    assert not (project.root / ".operation.lock").exists()
    assert "有限" in capsys.readouterr().err


def test_operation_lock_prevents_concurrent_mutation(project):
    with project.lock():
        with pytest.raises(LabError, match="另一个操作"):
            with project.lock():
                pass


def test_duplicate_pixels_are_not_added_twice(tmp_path):
    video = tmp_path / "static.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    for _ in range(10):
        writer.write(np.full((64, 64, 3), 100, np.uint8))
    writer.release()
    project = Project.create(tmp_path / "project", ["目标"])
    result = extract(project, video, every=0.2)
    assert result["added"] == 1
    assert result["duplicates_skipped"] == 9
    result = extract(project, video, every=0.2)
    assert result["added"] == 0


def test_no_positive_holdout_does_not_create_snapshot(project):
    for subset in ["seed", "val", "test"]:
        save_manual(project, subset, empty=subset == "test")
        import_labels(project, subset=subset)
    with pytest.raises(LabError, match="全负样本"):
        build_dataset(project, "invalid", [])
    assert not (project.root / "datasets" / "invalid").exists()


def test_invalid_seed_is_rejected_before_creating_project(tmp_path):
    with pytest.raises(LabError, match="随机种子"):
        Project.create(tmp_path / "bad", ["目标"], seed=-1)
    assert not (tmp_path / "bad").exists()


def test_nonfinite_end_is_rejected_before_extraction(tmp_path):
    project = Project.create(tmp_path / "project", ["目标"])
    video = tmp_path / "placeholder.avi"
    video.write_bytes(b"")
    with pytest.raises(LabError, match="结束时间"):
        extract(project, video, end=math.nan)
    assert project.frames() == []


def test_malformed_shape_flags_are_rejected(project):
    frame = project.frames("seed")[0]
    document = make_document(frame, [{"label": "目标", "points": [[10, 10], [20, 10], [20, 20], [10, 20]]}])
    document["shapes"][0]["flags"] = None
    with pytest.raises(LabError, match="flags"):
        parse_document(document, frame, ["目标"])


def test_status_is_available_during_a_running_operation(project, capsys):
    with project.lock():
        main(["--project", str(project.root), "status"])
    output = json.loads(capsys.readouterr().out)
    assert output["split_locked"] is True


def test_continue_plan_uses_parent_checkpoint(project):
    import_all(project)
    weights = project.root / "models" / "M0" / "best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"test parent checkpoint")
    write_json(weights.parent / "model.json", {"name": "M0", "status": "completed", "rounds": [],
               "model_yaml": "yolo26s-obb.yaml", "weights": "models/M0/best.pt", "weights_sha256": sha256(weights)})
    planned = train(project, "continued", init="continue", from_model="M0", dry_run=True)
    assert planned["params"]["pretrained"] is True
    assert planned["params"]["resume"] is False
    assert planned["parent_weights_sha256"] == sha256(weights)
    assert planned["from_model"] == "M0"
    assert planned["model_yaml"] == "yolo26s-obb.yaml"


@pytest.mark.parametrize("relative", ["images/train/copied_test.jpg", "images/val/nested/copied_test.PNG",
                                     "images/test/copied_test.bmp", "labels/train/orphan.txt",
                                     "images/train/stale.npy", "labels/train/nested/labels.cache"])
def test_snapshot_rejects_extra_input_files(project, relative):
    import_all(project)
    folder, _ = build_dataset(project, "M0", [])
    extra = folder / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    source = next((folder / "images" / "test").glob("*.jpg"))
    shutil.copy2(source, extra)
    with pytest.raises(LabError, match="文件集合不一致"):
        build_dataset(project, "M0", [])


def test_copying_test_pair_to_train_cannot_bypass_snapshot(project):
    import_all(project)
    folder, before = build_dataset(project, "M0", [])
    image = next((folder / "images" / "test").glob("*.jpg"))
    shutil.copy2(image, folder / "images" / "train" / image.name)
    label = folder / "labels" / "test" / f"{image.stem}.txt"
    shutil.copy2(label, folder / "labels" / "train" / label.name)
    with pytest.raises(LabError, match="文件集合不一致"):
        train(project, "M0", dry_run=True)
    assert before["counts"]["train"] == 4
    assert not (project.root / "reports" / "M0_plan.json").exists()


def test_snapshot_regenerates_disposable_label_caches(project):
    import_all(project)
    folder, before = build_dataset(project, "M0", [])
    for role in ["train", "val", "test"]:
        (folder / "labels" / f"{role}.cache").write_bytes(b"stale cached labels")
    _, after = build_dataset(project, "M0", [])
    assert before == after
    assert not list((folder / "labels").glob("*.cache"))


@pytest.mark.parametrize("boundary", ["archive", "index", "round"])
def test_review_import_recovers_from_each_failed_write(project, monkeypatch, boundary):
    import_all(project)
    frames = make_review_round(project)
    import autolabel_lab.dataset as dataset_module
    original_write = dataset_module.write_json
    failed = False

    def fail_once(path, value):
        nonlocal failed
        path = Path(path)
        trigger = ((boundary == "archive" and path.parent.name == "round_r1" and path.stem == frames[-1]["id"])
                   or (boundary == "index" and path.name == "annotations.json")
                   or (boundary == "round" and path.name == "round.json"))
        if trigger and not failed:
            failed = True
            raise OSError("simulated interrupted write")
        return original_write(path, value)

    monkeypatch.setattr(dataset_module, "write_json", fail_once)
    with pytest.raises(OSError, match="interrupted"):
        import_labels(project, round_name="r1", reviewed=True)
    assert failed
    assert project.round("r1")["status"] == "predicted"
    if boundary == "round":
        archived = copy.deepcopy(project.annotations["round_r1"])
        for frame in frames:
            # The recovery must use committed truth, not the now-changed source.
            (project.root / "rounds" / "r1" / "editable" / f"{frame['id']}.json").unlink()
    result = import_labels(project, round_name="r1", reviewed=True)
    assert project.round("r1")["status"] == "reviewed"
    assert project.round("r1")["reviewed_at"] == project.annotations["round_r1"]["imported_at"]
    assert len(project.require_annotations("round_r1")["frame_ids"]) == 2
    if boundary == "round":
        assert result["recovered"] is True
        assert project.annotations["round_r1"] == archived
    _, snapshot = build_dataset(project, "M1", ["r1"])
    assert snapshot["counts"]["train"] == 6


def test_review_retry_preserves_committed_labels(project):
    frames = make_review_round(project)
    import_labels(project, round_name="r1", reviewed=True)
    archived = copy.deepcopy(project.annotations["round_r1"])
    frame = frames[0]
    write_json(project.root / "rounds" / "r1" / "editable" / f"{frame['id']}.json", make_document(frame, []))
    result = import_labels(project, round_name="r1", reviewed=True)
    assert result["already_imported"] is True
    assert result["recovered"] is False
    assert project.annotations["round_r1"] == archived
    with pytest.raises(LabError, match="逐张审核"):
        import_labels(project, round_name="r1")


@pytest.mark.parametrize("damaged", ["archive", "raw", "frame_ids", "confirmation"])
def test_review_recovery_rejects_invalid_commit(project, monkeypatch, damaged):
    frames = make_review_round(project)
    import autolabel_lab.dataset as dataset_module
    original_write = dataset_module.write_json

    def fail_round_write(path, value):
        if Path(path).name == "round.json":
            raise OSError("simulated interrupted final write")
        return original_write(path, value)

    monkeypatch.setattr(dataset_module, "write_json", fail_round_write)
    with pytest.raises(OSError):
        import_labels(project, round_name="r1", reviewed=True)
    monkeypatch.setattr(dataset_module, "write_json", original_write)
    if damaged == "archive":
        (project.root / "annotations" / "round_r1" / f"{frames[0]['id']}.json").write_text("{}", encoding="utf-8")
    elif damaged == "raw":
        (project.root / "rounds" / "r1" / "raw" / f"{frames[0]['id']}.json").write_text("{}", encoding="utf-8")
    else:
        annotations = project.annotations
        if damaged == "frame_ids":
            annotations["round_r1"]["frame_ids"].pop()
        else:
            annotations["round_r1"]["human_review_confirmed"] = False
        write_json(project.root / "annotations.json", annotations)
    with pytest.raises(LabError):
        import_labels(project, round_name="r1", reviewed=True)
    assert project.round("r1")["status"] == "predicted"

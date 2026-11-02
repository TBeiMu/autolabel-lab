from __future__ import annotations

import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .labels import make_document, parse_document, yolo_text
from .cancellation import check_cancelled
from .project import LabError, fingerprint, now, read_json, safe_name, sha256, write_json


def sample_evenly(items, count):
    if count < 1 or count > len(items):
        raise LabError(f"需要 {count} 张图片，可用图片只有 {len(items)} 张。")
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, count).round().astype(int)]


def extract(project, video, every=1.0, start=0.0, end=None, timestamps=None, max_frames=1000):
    manifest = project.manifest
    if manifest["split"]:
        raise LabError("划分已固定；新增视频请建立新实验，避免改变验证/测试数据。")
    video = Path(video).resolve()
    if not video.is_file():
        raise LabError(f"视频不存在：{video}")
    if max_frames < 1 or not math.isfinite(every) or every <= 0 or not math.isfinite(start) or start < 0:
        raise LabError("抽帧间隔和上限必须大于 0，开始时间必须非负。")
    if end is not None and not math.isfinite(end):
        raise LabError("结束时间必须是有限秒数。")
    source_hash = sha256(video)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise LabError("OpenCV 无法打开视频，请检查编码与路径。")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise LabError("视频 FPS 或帧数不可用，请先转为恒定帧率视频。")
        duration = total / fps
        stop = min(duration, end) if end is not None else duration
        if not math.isfinite(stop) or stop <= start:
            raise LabError("结束时间必须大于开始时间，且抽帧范围需在视频内。")
        if timestamps is None:
            count = min(max_frames, math.ceil((stop - start) / every))
            times = [start + i * every for i in range(count)]
        else:
            times = sorted(set(timestamps))
            if not times or len(times) > max_frames or any(not math.isfinite(t) or t < start or t >= stop for t in times):
                raise LabError("指定时间点必须位于抽帧区间内，且不能超过 --max-frames。")
        indices = sorted({min(total - 1, int(round(t * fps))) for t in times})
        existing_indices = {(f["source_sha256"], f["frame_index"]) for f in manifest["frames"]}
        existing_hashes = {f["pixel_sha256"] for f in manifest["frames"]}
        added, duplicate = [], 0
        for index in indices:
            check_cancelled()
            if (source_hash, index) in existing_indices:
                duplicate += 1
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, pixels = capture.read()
            actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if not ok or actual != index:
                raise LabError(f"无法准确读取第 {index} 帧；已解码到 {actual}，请转为恒定帧率视频。")
            import hashlib
            pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
            if pixel_hash in existing_hashes:
                duplicate += 1
                continue
            frame_id = f"{source_hash[:12]}_f{index:09d}"
            path = project.root / "frames" / f"{frame_id}.jpg"
            ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise LabError("无法编码抽取图片。")
            path.write_bytes(encoded.tobytes())
            height, width = pixels.shape[:2]
            frame = {"id": frame_id, "file": path.name, "source": str(video), "source_sha256": source_hash,
                     "frame_index": index, "timestamp": index / fps, "fps": fps, "source_duration": duration,
                     "width": width, "height": height, "pixel_sha256": pixel_hash,
                     "image_sha256": sha256(path), "role": "unassigned"}
            added.append(frame)
            existing_hashes.add(pixel_hash)
        manifest["frames"].extend(added)
        write_json(project.root / "manifest.json", manifest)
        return {"added": len(added), "duplicates_skipped": duplicate, "video_duration_seconds": duration,
                "fps": fps, "total_extracted": len(manifest["frames"])}
    finally:
        capture.release()


def export_workspace(project, name, frames):
    folder = project.root / "workspaces" / name
    folder.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        shutil.copy2(project.image_path(frame), folder / frame["file"])
    (folder / "classes.txt").write_text("\n".join(project.config["classes"]) + "\n", encoding="utf-8")
    return folder


def split(project, seed_count=15, val_fraction=0.15, test_fraction=0.2, gap_seconds=2.0):
    manifest = project.manifest
    if manifest["split"]:
        raise LabError("数据划分已固定；不能重新划分，请建立新实验。")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise LabError("验证/测试比例必须大于 0，且总和小于 1。")
    if not math.isfinite(gap_seconds) or gap_seconds < 0:
        raise LabError("时间隔离间隔必须是非负有限数。")
    sources = sorted({f["source_sha256"] for f in manifest["frames"]})
    if not sources:
        raise LabError("请先抽帧。")
    boundaries = []
    for source in sources:
        frames = sorted([f for f in manifest["frames"] if f["source_sha256"] == source], key=lambda f: f["timestamp"])
        n_val, n_test = max(1, math.ceil(len(frames) * val_fraction)), max(1, math.ceil(len(frames) * test_fraction))
        a, b = len(frames) - n_val - n_test, len(frames) - n_test
        if a < 1:
            raise LabError("每个视频至少需要足够的抽帧供训练、验证和测试。")
        cut1 = (frames[a - 1]["timestamp"] + frames[a]["timestamp"]) / 2
        cut2 = (frames[b - 1]["timestamp"] + frames[b]["timestamp"]) / 2
        boundaries.append({"source_sha256": source, "train_val_boundary": cut1, "val_test_boundary": cut2})
        for frame in frames:
            t = frame["timestamp"]
            frame["role"] = "pool" if t < cut1 else "val" if t < cut2 else "test"
            if min(abs(t - cut1), abs(t - cut2)) < gap_seconds / 2:
                frame["role"] = "gap"
    train = sorted([f for f in manifest["frames"] if f["role"] == "pool"], key=lambda f: (f["source_sha256"], f["timestamp"]))
    for frame in sample_evenly(train, seed_count):
        frame["role"] = "seed"
    if not all(any(f["role"] == role for f in manifest["frames"]) for role in ["val", "test"]):
        raise LabError("隔离后验证/测试集为空，请增加抽帧或缩小 --gap-seconds。")
    manifest["split"] = {"created_at": now(), "method": "chronological_per_video", "seed_count": seed_count,
                         "val_fraction": val_fraction, "test_fraction": test_fraction, "gap_seconds": gap_seconds,
                         "boundaries": boundaries, "frames_hash": fingerprint(manifest["frames"])}
    # Copy all editable images before committing the frozen manifest.
    for role in ["seed", "val", "test"]:
        export_workspace(project, role, [f for f in manifest["frames"] if f["role"] == role])
    write_json(project.root / "manifest.json", manifest)
    return {"counts": {r: sum(f["role"] == r for f in manifest["frames"]) for r in ["seed", "pool", "val", "test", "gap"]},
            "workspaces": str(project.root / "workspaces")}


def import_labels(project, subset=None, round_name=None, source=None, reviewed=False):
    project.require_split()
    if round_name:
        round_name = safe_name(round_name)
        record = project.round(round_name)
        if record["status"] not in {"predicted", "reviewed"}:
            raise LabError("此轮尚未预测成功，不能导入。")
        if not reviewed:
            raise LabError("请逐张审核（包括无预测的图片），再使用 --reviewed 确认。")
        target = f"round_{round_name}"
        frame_ids = set(record["frame_ids"])
        frames = [f for f in project.frames() if f["id"] in frame_ids]
        if len(frame_ids) != len(record["frame_ids"]) or len(frames) != len(frame_ids) or any(f["role"] != "pool" for f in frames):
            raise LabError("轮次图片清单必须是无重复的训练池图片。")
        default_source = project.root / "rounds" / round_name / "editable"
    else:
        if subset not in {"seed", "val", "test"}:
            raise LabError("人工标注子集必须是 seed、val 或 test。")
        target = subset
        frames = project.frames(subset)
        default_source = project.root / "workspaces" / subset
    if target in project.annotations:
        if not round_name:
            raise LabError("这份人工标注已固定归档，不能覆盖。")
        # annotations.json is the commit point. Recover from the archive, even if
        # editable files have since changed or disappeared after a failed final write.
        annotation = project.require_annotations(target)
        if (not annotation.get("human_review_confirmed")
                or set(annotation["frame_ids"]) != frame_ids
                or len(annotation["frame_ids"]) != len(frame_ids)
                or set(annotation["hashes"]) != frame_ids
                or annotation["folder"] != f"annotations/{target}"):
            raise LabError("已归档标注与审核轮次不一致，拒绝恢复。")
        for frame in frames:
            raw = project.root / "rounds" / round_name / "raw" / f"{frame['id']}.json"
            if not raw.is_file() or sha256(raw) != record["raw_hashes"].get(frame["id"]):
                raise LabError(f"原始预测缺失或被修改：{raw}")
        if record["status"] == "reviewed" and record.get("annotation_key") != target:
            raise LabError("审核轮次的归档引用不一致，拒绝恢复。")
        recovered = record["status"] == "predicted"
        if recovered:
            record.update(status="reviewed", reviewed_at=annotation["imported_at"], annotation_key=target)
            write_json(project.root / "rounds" / round_name / "round.json", record)
        return {"subset": target, "frames": len(frames), "boxes": annotation["boxes"],
                "archive": str(project.root / annotation["folder"]), "already_imported": True, "recovered": recovered}
    if round_name and record["status"] == "reviewed":
        raise LabError("已审核轮次缺少归档标注，不能通过重新导入覆盖。")
    source = Path(source).resolve() if source else default_source
    prepared = []
    for frame in frames:
        path = source / f"{frame['id']}.json"
        if not path.is_file():
            raise LabError(f"缺少人工标注：{path}。无目标帧也必须明确保存空 shapes JSON。")
        image = source / frame["file"]
        if image.is_file() and sha256(image) != frame["image_sha256"]:
            raise LabError(f"编辑目录中的图片被替换或重新编码：{image}")
        document = read_json(path)
        if not isinstance(document, dict) or not isinstance(document.get("flags", {}), dict):
            raise LabError(f"标注 JSON 和 flags 必须是对象：{path}")
        if document.get("flags", {}).get("autolabel_lab_prediction") and not reviewed:
            raise LabError("检测到模型预测，请人工审核后使用 --reviewed。")
        shapes = parse_document(document, frame, project.config["classes"])
        # Confidence is a model measurement, not a property of human truth.
        for shape in shapes:
            shape.pop("score", None)
        prepared.append((frame, make_document(frame, shapes)))
    folder = project.root / "annotations" / target
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for frame, document in prepared:
        path = folder / f"{frame['id']}.json"
        write_json(path, document)
        hashes[frame["id"]] = sha256(path)
    annotations = project.annotations
    annotations[target] = {"folder": folder.relative_to(project.root).as_posix(), "hashes": hashes,
                           "imported_at": now(), "frame_ids": [f["id"] for f, _ in prepared],
                           "boxes": sum(len(d["shapes"]) for _, d in prepared), "human_review_confirmed": bool(reviewed)}
    write_json(project.root / "annotations.json", annotations)
    if round_name:
        record.update(status="reviewed", reviewed_at=annotations[target]["imported_at"], annotation_key=target)
        write_json(project.root / "rounds" / round_name / "round.json", record)
    return {"subset": target, "frames": len(prepared), "boxes": annotations[target]["boxes"], "archive": str(folder)}


def validate_snapshot(folder, snapshot):
    expected = {p for p in snapshot["file_hashes"] if p.startswith(("images/", "labels/"))}
    # Ultralytics stores label caches beside split directories. They are disposable
    # derivatives, not trusted labels; image .npy caches inside splits are rejected.
    caches = {f"labels/{role}.cache" for role in ["train", "val", "test"]}
    actual = {path.relative_to(folder).as_posix()
              for directory in ["images", "labels"] for path in (folder / directory).rglob("*") if path.is_file()}
    if actual - caches != expected:
        extra = sorted((actual - caches) - expected)
        missing = sorted(expected - actual)
        raise LabError(f"数据快照文件集合不一致：新增 {extra}；缺失 {missing}。")
    for relative, digest in snapshot["file_hashes"].items():
        if not (folder / relative).is_file() or sha256(folder / relative) != digest:
            raise LabError(f"数据快照被修改：{folder / relative}")
    # Cached labels can diverge from the verified text files. Force the consumer to
    # reconstruct caches from those files on each training/evaluation operation.
    for relative in caches:
        (folder / relative).unlink(missing_ok=True)


def build_dataset(project, name, rounds):
    project.require_split()
    rounds = list(dict.fromkeys(safe_name(r) for r in rounds))
    keys = ["seed", "val", "test"]
    for r in rounds:
        record = project.round(r)
        if record["status"] != "reviewed":
            raise LabError(f"{r} 尚未完成人工审核。")
        keys.append(record["annotation_key"])
    info = {key: project.require_annotations(key) for key in keys}
    train_ids = [i for key in keys if key not in {"val", "test"} for i in info[key]["frame_ids"]]
    if len(train_ids) != len(set(train_ids)):
        raise LabError("训练数据包含重复图片。")
    holdout_ids = set(info["val"]["frame_ids"] + info["test"]["frame_ids"])
    if set(train_ids) & holdout_ids:
        raise LabError("验证/测试图片进入训练集，拒绝训练。")
    if not train_ids or any(not info[role]["frame_ids"] for role in ["val", "test"]):
        raise LabError("训练、验证和测试集都必须有图片。")
    # Validate class coverage before writing a new snapshot, so errors leave no partial dataset.
    prepared = {}
    class_counts = {role: {c: 0 for c in project.config["classes"]} for role in ["train", "val", "test"]}
    frames_by_id = {f["id"]: f for f in project.frames()}
    for key, annotation in info.items():
        role = key if key in {"val", "test"} else "train"
        for frame_id in annotation["frame_ids"]:
            frame = frames_by_id[frame_id]
            shapes = parse_document(read_json(project.root / annotation["folder"] / f"{frame_id}.json"), frame, project.config["classes"])
            prepared[frame_id] = shapes
            for shape in shapes:
                class_counts[role][shape["label"]] += 1
    missing = [c for c, n in class_counts["train"].items() if n == 0]
    if missing:
        raise LabError(f"训练数据没有覆盖类别：{missing}，请补充种子数据并新建实验。")
    if any(sum(class_counts[role].values()) == 0 for role in ["val", "test"]):
        raise LabError("验证/测试集需要至少一个人工标注目标；全负样本无法比较检测 mAP。")
    description = {"classes": project.config["classes"], "split": project.manifest["split"],
                   "annotation_hashes": {k: v["hashes"] for k, v in info.items()}, "rounds": rounds}
    data_hash = fingerprint(description)
    folder = project.root / "datasets" / safe_name(name)
    meta_path = folder / "snapshot.json"
    if folder.exists():
        if not meta_path.is_file() or read_json(meta_path)["dataset_hash"] != data_hash:
            raise LabError("同名数据快照已存在且内容不一致，请换一个名称。")
        saved = read_json(meta_path)
        validate_snapshot(folder, saved)
        return folder, saved
    folder.mkdir(parents=True)
    file_hashes = {}
    counts, objects = {k: 0 for k in ["train", "val", "test"]}, {k: 0 for k in project.config["classes"]}
    for key, annotation in info.items():
        role = key if key in {"val", "test"} else "train"
        for frame_id in annotation["frame_ids"]:
            frame = frames_by_id[frame_id]
            shapes = prepared[frame_id]
            image = folder / "images" / role / frame["file"]
            label = folder / "labels" / role / f"{frame_id}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project.image_path(frame), image)
            label.write_text(yolo_text(frame, shapes, project.config["classes"]), encoding="utf-8")
            file_hashes[image.relative_to(folder).as_posix()] = sha256(image)
            file_hashes[label.relative_to(folder).as_posix()] = sha256(label)
            counts[role] += 1
            if role == "train":
                for shape in shapes:
                    objects[shape["label"]] += 1
    data = {"path": str(folder), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": dict(enumerate(project.config["classes"]))}
    yaml_path = folder / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    file_hashes["data.yaml"] = sha256(yaml_path)
    test_hash = fingerprint({"classes": project.config["classes"], "test_images": {f["id"]: f["image_sha256"] for f in project.frames("test")},
                             "test_labels": info["test"]["hashes"]})
    snapshot = {**description, "dataset_hash": data_hash, "test_hash": test_hash,
                "counts": counts, "train_class_boxes": objects, "class_boxes_by_split": class_counts,
                "file_hashes": file_hashes, "created_at": now()}
    write_json(meta_path, snapshot)
    return folder, snapshot

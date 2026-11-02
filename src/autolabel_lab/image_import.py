"""Import an existing YOLO OBB dataset without changing its images or labels."""
from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from .cancellation import check_cancelled
from .dataset import export_workspace, import_labels
from .labels import make_document, quad
from .project import LabError, fingerprint, now, sha256, write_json


def dataset_classes(source):
    root = Path(source).resolve()
    try:
        config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8-sig"))
        names = config["names"]
        if isinstance(names, dict):
            normalized = {int(k): v for k, v in names.items()}
            if set(normalized) != set(range(len(normalized))) or len(normalized) != len(names):
                raise ValueError("类别编号必须从 0 连续排列")
            names = [normalized[i] for i in range(len(normalized))]
        if not isinstance(names, list) or not names or any(not isinstance(c, str) or not c.strip() or c != c.strip() for c in names):
            raise ValueError("names 必须是非空类别名称清单")
        if len(set(names)) != len(names) or config.get("nc", len(names)) != len(names):
            raise ValueError("类别名称重复或 nc 不一致")
        return names
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise LabError(f"无法读取 YOLO 数据集类别：{root / 'data.yaml'}：{exc}") from exc


def read_sample(root, image, classes):
    relative = image.relative_to(root / "images")
    label = root / "labels" / relative.with_suffix(".txt")
    if not label.is_file():
        raise LabError(f"缺少标注：{label}。无目标图片也需要空 TXT。")
    data = image.read_bytes()
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise LabError(f"无法读取图片：{image}")
    height, width = pixels.shape[:2]
    shapes = []
    for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = line.split()
            if len(values) != 9:
                raise ValueError("OBB 每行需要类别编号和四角的 8 个归一化坐标")
            index = int(values[0])
            if not 0 <= index < len(classes):
                raise ValueError("未知类别编号")
            points = np.asarray([float(v) for v in values[1:]]).reshape(4, 2)
            shapes.append({"label": classes[index], "points": quad(points * [width, height], width, height)})
        except (ValueError, LabError) as exc:
            raise LabError(f"{label} 第 {line_number} 行：{exc}") from exc
    pixel_hash = fingerprint({"width": width, "height": height, "pixels": hashlib.sha256(pixels.tobytes()).hexdigest()})
    match = re.match(r"(2_annotations_[A-D]_view[123]|1_improve_\d+)", image.stem)
    group = match.group(1) if match else re.sub(r"(?:_frame)?_\d+$", "", image.stem)
    return {"image": image, "label": label, "relative": image.relative_to(root).as_posix(),
            "original_split": relative.parts[0], "width": width, "height": height,
            "image_sha256": hashlib.sha256(data).hexdigest(), "label_sha256": sha256(label),
            "pixel_sha256": pixel_hash, "shapes": shapes, "group": group}


def select_samples(samples, count, rng, classes):
    if count > len(samples):
        raise LabError(f"需要 {count} 张，可用独立图片只有 {len(samples)} 张。")
    # Choose using a fixed seed, and require every class in each selected subset.
    for _ in range(1000):
        selected = rng.sample(samples, count)
        present = {s["label"] for item in selected for s in item["shapes"]}
        if present == set(classes):
            return selected
    raise LabError(f"{count} 张无法覆盖所有类别，请增加图片数或补充标注。")


def import_yolo(project, source, train_count=200, val_count=50, pool_count=200, test_count=50):
    if project.manifest["frames"] or project.manifest["split"]:
        raise LabError("图片导入需要空实验；已有数据请新建实验。")
    counts = {"seed": train_count, "val": val_count, "pool": pool_count, "test": test_count}
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in counts.values()):
        raise LabError("训练、验证、审核池、固定测试图片数必须为正整数。")
    root = Path(source).resolve()
    classes = dataset_classes(root)
    if classes != project.config["classes"]:
        raise LabError("数据集类别名称或顺序与实验不一致。")
    samples, pixels, duplicates = [], set(), []
    for image in sorted((root / "images").rglob("*")):
        check_cancelled()
        if not image.is_file() or image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        item = read_sample(root, image, classes)
        if item["pixel_sha256"] in pixels:
            duplicates.append(item["relative"])
            continue
        pixels.add(item["pixel_sha256"])
        samples.append(item)
    rng = random.Random(project.config["seed"])
    val_source = [s for s in samples if s["original_split"] == "val"]
    candidates = [s for s in samples if s["original_split"] == "train"]
    if not val_source or not candidates:
        raise LabError("需要 images/train、images/val 和配套 labels/train、labels/val。data.yaml 的旧 path 不参与路径解析。")
    selected = {"val": select_samples(val_source, val_count, rng, classes)}
    for role in ["test", "seed", "pool"]:
        selected[role] = select_samples(candidates, counts[role], rng, classes)
        chosen = {item["relative"] for item in selected[role]}
        candidates = [item for item in candidates if item["relative"] not in chosen]
    source_id = fingerprint({"source": str(root)})
    frames, records, class_counts = [], [], {}
    for role in ["seed", "val", "test", "pool"]:
        class_counts[role] = dict(Counter(s["label"] for item in selected[role] for s in item["shapes"]))
        for item in sorted(selected[role], key=lambda s: s["relative"]):
            check_cancelled()
            frame_id = "img_" + fingerprint({"source": item["relative"], "sha256": item["image_sha256"]})[:24]
            frame = {"id": frame_id, "file": frame_id + item["image"].suffix.lower(),
                     "source": str(item["image"]), "source_sha256": source_id, "source_kind": "image",
                     "source_name": item["image"].name, "frame_index": len(frames), "timestamp": None,
                     "width": item["width"], "height": item["height"], "pixel_sha256": item["pixel_sha256"],
                     "image_sha256": item["image_sha256"], "role": role}
            if sha256(item["image"]) != item["image_sha256"] or sha256(item["label"]) != item["label_sha256"]:
                raise LabError("源数据在检查后发生变化，请重试。")
            shutil.copy2(item["image"], project.root / "frames" / frame["file"])
            folder = project.root / ("references/pool" if role == "pool" else f"workspaces/{role}")
            write_json(folder / f"{frame_id}.json", make_document(frame, item["shapes"]))
            records.append({"id": frame_id, "role": role, "source_image": str(item["image"]),
                            "source_label": str(item["label"]), "source_relative": item["relative"],
                            "source_label_sha256": item["label_sha256"], "image_sha256": item["image_sha256"],
                            "group": item["group"], "original_split": item["original_split"]})
            frames.append(frame)
    for role in ["seed", "val", "test"]:
        export_workspace(project, role, [f for f in frames if f["role"] == role])
    split = {"created_at": now(), "method": "imported_yolo_fixed_seed", "seed_count": train_count,
             "counts": counts, "frames_hash": fingerprint(frames), "random_seed": project.config["seed"],
             "val_source": "original_val", "holdout_source": "original_train"}
    write_json(project.root / "manifest.json", {"frames": frames, "split": split})
    for role in ["seed", "val", "test"]:
        import_labels(project, subset=role)
    warnings = ["数据已打乱，按图片抽样；同一动作/视频/视角可能跨子集，结果不能证明对独立新场景的泛化。",
                "已有标签作为参考答案导入，本轮未重新人工审查；标签质量需按实际需要抽查。",
                "审核池参考标注仅保存在 references/pool，不导入训练、不复制到预测审核目录。"]
    provenance = {"source": str(root), "source_yaml_sha256": sha256(root / "data.yaml"),
                  "total_unique_images": len(samples), "selected_counts": counts,
                  "unused_images": len(samples) - len(frames), "duplicates_skipped": duplicates,
                  "class_boxes": class_counts, "records": records, "warnings": warnings, "created_at": now()}
    write_json(project.root / "source_import.json", provenance)
    project.config["input_kind"] = "yolo_images"
    project.config["import_warnings"] = warnings
    write_json(project.root / "project.json", project.config)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    (project.root / "reports" / "selection.csv").write_text(stream.getvalue(), encoding="utf-8-sig")
    return {"counts": counts, "unused_images": provenance["unused_images"], "classes": classes,
            "class_boxes": class_counts, "warnings": warnings, "selection": str(project.root / "reports/selection.csv")}

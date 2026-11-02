from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .project import LabError


def quad(points, width, height, check_bounds=True):
    try:
        array = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise LabError("旋转框坐标必须是数字。") from exc
    if array.shape != (4, 2) or not np.isfinite(array).all():
        raise LabError("旋转框必须包含四个有限二维坐标。")
    if check_bounds and ((array < 0).any() or (array[:, 0] > width).any() or (array[:, 1] > height).any()):
        raise LabError("旋转框超出图片边界，请在 X-AnyLabeling 中修正后导入。")
    hull = cv2.convexHull(array.astype(np.float32)).reshape(-1, 2)
    if len(hull) != 4 or abs(cv2.contourArea(hull)) < 1:
        raise LabError("旋转框退化、重复点或不是凸四边形。")
    # Keep edge direction stable while allowing cyclic corner order from different tools.
    start = min(range(4), key=lambda i: (float(hull[i, 1]), float(hull[i, 0])))
    hull = np.roll(hull, -start, axis=0)
    return hull.astype(float).tolist()


def parse_document(document, frame, classes, check_bounds=True):
    if not isinstance(document, dict) or not isinstance(document.get("shapes"), list):
        raise LabError("标注 JSON 必须包含 shapes 列表；未标注图片不能当作负样本。")
    if Path(str(document.get("imagePath", "")).replace("\\", "/")).name != frame["file"]:
        raise LabError("标注 imagePath 与对应原图不一致。")
    if document.get("imageWidth") != frame["width"] or document.get("imageHeight") != frame["height"]:
        raise LabError("标注图片尺寸与原图不一致。")
    shapes = []
    for shape in document["shapes"]:
        if not isinstance(shape, dict) or shape.get("label") not in classes:
            raise LabError(f"未知类别：{shape.get('label') if isinstance(shape, dict) else shape}。")
        if not isinstance(shape.get("flags", {}), dict):
            raise LabError("shape flags 必须是对象。")
        if shape.get("difficult", False) or shape.get("flags", {}).get("ignore", False):
            raise LabError("本工具不支持 difficult/ignore 框，请明确其处理方式后再导入。")
        points = shape.get("points")
        shape_type = shape.get("shape_type")
        if shape_type == "rectangle" and isinstance(points, list) and len(points) == 2:
            try:
                (x1, y1), (x2, y2) = points
            except (TypeError, ValueError) as exc:
                raise LabError("矩形必须包含两个二维坐标。") from exc
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif shape_type not in {"rotation", "polygon", "rectangle"}:
            raise LabError(f"不支持 {shape_type}，请使用旋转框（rotation）。")
        parsed = {"label": shape["label"], "points": quad(points, frame["width"], frame["height"], check_bounds)}
        score = shape.get("score")
        if score is not None:
            try:
                score = float(score)
            except (ValueError, TypeError) as exc:
                raise LabError("预测置信度必须是数字。") from exc
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise LabError("预测置信度必须在 0 到 1 之间。")
            parsed["score"] = score
        shapes.append(parsed)
    return shapes


def make_document(frame, shapes, prediction=False):
    exported = []
    for shape in shapes:
        points = shape["points"]
        dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
        exported.append({"label": shape["label"], "points": points, "group_id": None,
                         "shape_type": "rotation", "flags": {}, "difficult": False,
                         "direction": math.atan2(dy, dx) % (2 * math.pi), "description": "",
                         **({"score": shape["score"]} if "score" in shape else {})})
    return {"version": "3.0.0", "flags": {"autolabel_lab_prediction": prediction}, "shapes": exported,
            "imagePath": frame["file"], "imageData": None, "imageHeight": frame["height"], "imageWidth": frame["width"]}


def yolo_text(frame, shapes, classes):
    rows = []
    for shape in shapes:
        points = quad(shape["points"], frame["width"], frame["height"])
        coordinates = [value / (frame["width"] if i % 2 == 0 else frame["height"])
                       for i, value in enumerate(np.asarray(points).reshape(-1))]
        rows.append(str(classes.index(shape["label"])) + " " + " ".join(f"{v:.8f}" for v in coordinates))
    return "\n".join(rows) + ("\n" if rows else "")


def polygon_iou(left, right):
    a, b = np.asarray(left, np.float32), np.asarray(right, np.float32)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - intersection
    return max(0.0, min(1.0, float(intersection / union))) if union > 0 else 0.0


def compare_shapes(predictions, truth, match_iou=0.5, unchanged_iou=0.95):
    # First match correct classes by descending geometric IoU, then detect class corrections.
    remaining_p, remaining_t = set(range(len(predictions))), set(range(len(truth)))
    counts = {"unchanged": 0, "adjusted": 0, "class_changed": 0, "false_positive": 0, "missed": 0}
    pairs = sorted(((polygon_iou(p["points"], t["points"]), pi, ti)
                    for pi, p in enumerate(predictions) for ti, t in enumerate(truth)), reverse=True)
    matched_ious = []
    for same_class in (True, False):
        for iou, pi, ti in pairs:
            if iou < match_iou or pi not in remaining_p or ti not in remaining_t:
                continue
            same = predictions[pi]["label"] == truth[ti]["label"]
            if same != same_class:
                continue
            remaining_p.remove(pi)
            remaining_t.remove(ti)
            if not same:
                counts["class_changed"] += 1
            else:
                counts["unchanged" if iou >= unchanged_iou else "adjusted"] += 1
                matched_ious.append(iou)
    counts["false_positive"], counts["missed"] = len(remaining_p), len(remaining_t)
    return {**counts, "prediction_boxes": len(predictions), "truth_boxes": len(truth), "matched_ious": matched_ious}

from __future__ import annotations

import csv
import io
import math

from .labels import compare_shapes, parse_document
from .project import LabError, now, read_json, sha256, write_json


def record_time(project, stage, seconds, round_name=None, frame_id=None, note=""):
    project.require_split()
    if not math.isfinite(seconds) or seconds <= 0:
        raise LabError("人工作时必须是大于 0 的有限秒数。")
    if stage == "seed":
        if round_name:
            raise LabError("种子标注工时不应关联自动标注轮次。")
        ids = [f["id"] for f in project.frames("seed")]
    else:
        if not round_name:
            raise LabError("review/manual 工时需要 --round。")
        record = project.round(round_name)
        if record["status"] not in {"predicted", "reviewed"}:
            raise LabError("该轮没有成功的预测。")
        ids = record["frame_ids"]
    if frame_id:
        if frame_id not in ids:
            raise LabError("此图片不属于所选标注阶段。")
        ids = [frame_id]
    timings = read_json(project.root / "timings.json")
    item = {"stage": stage, "round": round_name, "frame_ids": ids, "seconds": seconds, "note": note, "created_at": now()}
    timings.append(item)
    write_json(project.root / "timings.json", timings)
    return item


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def round_report(project, record, match_iou, unchanged_iou, timings):
    key = record["annotation_key"]
    annotation = project.require_annotations(key)
    frames = {f["id"]: f for f in project.frames()}
    counts = {key: 0 for key in ["unchanged", "adjusted", "class_changed", "false_positive", "missed", "prediction_boxes", "truth_boxes"]}
    per_image, ious = [], []
    for frame_id in record["frame_ids"]:
        frame = frames[frame_id]
        path = project.root / "rounds" / record["name"] / "raw" / f"{frame_id}.json"
        if not path.is_file() or sha256(path) != record["raw_hashes"][frame_id]:
            raise LabError(f"原始预测缺失或被修改：{path}")
        predictions = parse_document(read_json(path), frame, project.config["classes"], check_bounds=False)
        truth = parse_document(read_json(project.root / annotation["folder"] / f"{frame_id}.json"), frame, project.config["classes"])
        measured = compare_shapes(predictions, truth, match_iou, unchanged_iou)
        ious.extend(measured.pop("matched_ious"))
        for key in counts:
            counts[key] += measured[key]
        per_image.append({"frame": frame_id, **measured})
    tp = counts["unchanged"] + counts["adjusted"]
    result = {"name": record["name"], "model": record["model"], "frames": len(per_image), "params": record["params"],
              "counts": counts, "precision_at_match_iou": ratio(tp, counts["prediction_boxes"]),
              "recall_at_match_iou": ratio(tp, counts["truth_boxes"]), "mean_matched_iou": ratio(sum(ious), len(ious)),
              "machine_seconds": record.get("machine_seconds"), "per_image": per_image}
    frame_set = set(record["frame_ids"])
    time_result = {}
    for stage in ["manual", "review"]:
        entries = [t for t in timings if t["round"] == record["name"] and t["stage"] == stage]
        coverage = set(i for t in entries for i in t["frame_ids"])
        time_result[stage] = {"seconds": sum(t["seconds"] for t in entries), "frames_covered": len(coverage),
                              "complete": bool(entries) and coverage == frame_set}
    if all(time_result[s]["complete"] for s in ["manual", "review"]):
        time_result["reported_time_reduction"] = 1 - time_result["review"]["seconds"] / time_result["manual"]["seconds"]
    else:
        time_result["reported_time_reduction"] = None
    result["human_time"] = time_result
    return result


def fmt(value):
    return "未记录" if value is None else f"{value:.4f}"


def report(project, match_iou=0.5, unchanged_iou=0.95):
    project.require_split()
    if not (0 < match_iou <= unchanged_iou <= 1):
        raise LabError("需要 0 < match-iou <= unchanged-iou <= 1。")
    timings = read_json(project.root / "timings.json")
    rounds = []
    for path in sorted((project.root / "rounds").glob("*/round.json")):
        record = read_json(path)
        if record["status"] == "reviewed":
            rounds.append(round_report(project, record, match_iou, unchanged_iou, timings))
    models = []
    for path in (project.root / "models").glob("*/model.json"):
        record = read_json(path)
        if record["status"] == "completed":
            project.model(record["name"])
            eval_path = path.parent / "evaluation.json"
            if eval_path.is_file():
                evaluation = read_json(eval_path)
                if evaluation["test_hash"] != record["test_hash"] or evaluation["weights_sha256"] != record["weights_sha256"]:
                    raise LabError("评价记录与模型不匹配。")
                record["evaluation"] = evaluation
            models.append(record)
    models.sort(key=lambda m: m["created_at"])
    warnings = ["此实验按单个视频内的连续时间段隔离；不能据此证明对新视频或新场景的泛化。",
                "修正分类来自四边形 IoU 的贪心匹配。类别一致且 IoU 高仅代表几何近似，不代表人工没有操作。",
                "审核结果是本轮参考标注；漏检统计依赖审核人补齐所有目标，应抽样复核。",
                "纯人工与审核工时由用户记录。使用不同标注人或交叉顺序控制记忆/熟练度，不把估计耗时当实测。",
                "测试指标的 P/R 来自 Ultralytics 评价逻辑，和本轮匹配阈值的 P/R 定义不同。"]
    evaluated = [m for m in models if "evaluation" in m]
    comparison = None
    if len(evaluated) >= 2:
        first, last = evaluated[0], evaluated[-1]
        if first["test_hash"] != last["test_hash"]:
            warnings.append("模型测试集不同，不计算指标差值。")
        else:
            comparison = {"baseline": first["name"], "latest": last["name"],
                          "deltas": {k: last["evaluation"]["metrics"][k] - v for k, v in first["evaluation"]["metrics"].items()}}
        settings = lambda m: {"model_yaml": m["model_yaml"], "params": {k: v for k, v in m["params"].items() if k != "device"}}
        if any(m["init"] != "scratch" for m in evaluated):
            warnings.append("含继续训练模型，提升同时受新增数据和额外训练影响；不能只归因于自动标注。")
        if any(settings(m) != settings(first) for m in evaluated[1:]):
            warnings.append("模型训练配置不同，模型差值不构成严格的数据增量对照。")
    for role in ["val", "test"]:
        if role in project.annotations:
            annotation = project.require_annotations(role)
            present = set()
            for frame_id in annotation["frame_ids"]:
                document = read_json(project.root / annotation["folder"] / f"{frame_id}.json")
                present.update(s["label"] for s in document["shapes"])
            absent = [c for c in project.config["classes"] if c not in present]
            if absent:
                warnings.append(f"{role} 没有类别 {absent} 的人工目标，不能评价这些类别的效果。")
    synthetic = bool(project.config.get("synthetic_smoke_test", False))
    report_data = {"created_at": now(), "synthetic_smoke_test": synthetic, "classes": project.config["classes"], "split": project.manifest["split"],
                   "match_iou": match_iou, "unchanged_iou": unchanged_iou, "rounds": rounds, "models": models,
                   "comparison": comparison, "seed_human_seconds": sum(t["seconds"] for t in timings if t["stage"] == "seed"),
                   "warnings": warnings}
    folder = project.root / "reports"
    write_json(folder / "report.json", report_data)
    lines = ["# 自动标注实验报告", "", "类别：" + "、".join(project.config["classes"]), "",
             f"标注比较：匹配 IoU ≥ {match_iou}；几何近似 IoU ≥ {unchanged_iou}。", "",
             "## 预标注与审核标注的差异", "", "| 轮次 | 模型 | 图片 | 几何近似 | 框调整 | 改类别 | 多余预测 | 补漏 | P | R |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if synthetic:
        lines[0:0] = ["> 本报告来自模拟视频的工具验证；不能作为真实自动标注实验结果。", ""]
    for r in rounds:
        c = r["counts"]
        lines.append(f"| {r['name']} | {r['model']} | {r['frames']} | {c['unchanged']} | {c['adjusted']} | {c['class_changed']} | {c['false_positive']} | {c['missed']} | {fmt(r['precision_at_match_iou'])} | {fmt(r['recall_at_match_iou'])} |")
    if not rounds:
        lines.extend(["", "尚无已审核轮次；不会把原始预测当成人工参考答案。"])
    lines.extend(["", "## 人工作时", "", f"种子标注已记录：{report_data['seed_human_seconds'] / 60:.2f} 分钟。", "",
                  "| 轮次 | 纯人工基线/分钟 | 审核修正/分钟 | 已记录耗时减少比例 |", "|---|---:|---:|---:|"])
    for r in rounds:
        t = r["human_time"]
        baseline = f"{t['manual']['seconds']/60:.2f}" if t["manual"]["complete"] else "缺少完整工时"
        review = f"{t['review']['seconds']/60:.2f}" if t["review"]["complete"] else "缺少完整工时"
        saving = f"{t['reported_time_reduction']:.1%}" if t["reported_time_reduction"] is not None else "不能计算"
        lines.append(f"| {r['name']} | {baseline} | {review} | {saving} |")
    lines.extend(["", "## 固定测试集模型指标", "", "| 模型 | 初始化 | 训练图片 | P | R | mAP50 | mAP50–95 | 机器训练/分钟 |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for m in models:
        metrics = m.get("evaluation", {}).get("metrics", {})
        lines.append(f"| {m['name']} | {m['init']} | {m['counts']['train']} | {fmt(metrics.get('precision'))} | {fmt(metrics.get('recall'))} | {fmt(metrics.get('mAP50'))} | {fmt(metrics.get('mAP50_95'))} | {m['machine_seconds']/60:.2f} |")
    if comparison:
        lines.extend(["", f"{comparison['latest']} 相对 {comparison['baseline']}：" + "；".join(f"{k} {v:+.4f}" for k, v in comparison["deltas"].items()) + "。"])
    lines.extend(["", "## 解释边界", "", *[f"- {w}" for w in warnings], "",
                  "- 模型提升表示新增的合格训练数据有用；自动标注的价值还需用合格标注的人工作时验证。",
                  "- 相同 epochs 下，增加数据也会增加训练更新次数。本工具记录训练配置与机器耗时。", ""])
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    stream = io.StringIO(newline="")
    fieldnames = ["round", "frame", "unchanged", "adjusted", "class_changed", "false_positive", "missed", "prediction_boxes", "truth_boxes"]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for r in rounds:
        for item in r["per_image"]:
            writer.writerow({"round": r["name"], **item})
    (folder / "per_image.csv").write_text(stream.getvalue(), encoding="utf-8-sig")
    return {"report": str(folder / "report.md"), "json": str(folder / "report.json"),
            "per_image_csv": str(folder / "per_image.csv"), "reviewed_rounds": len(rounds), "evaluated_models": len(evaluated)}


def status(project):
    manifest = project.manifest
    rounds = [read_json(p) for p in sorted((project.root / "rounds").glob("*/round.json"))]
    models = [read_json(p) for p in sorted((project.root / "models").glob("*/model.json"))]
    return {"project": str(project.root), "classes": project.config["classes"], "split_locked": bool(manifest["split"]),
            "frames": {role: sum(f["role"] == role for f in manifest["frames"]) for role in ["unassigned", "seed", "pool", "val", "test", "gap"]},
            "annotations": {k: {"frames": len(v["frame_ids"]), "boxes": v["boxes"]} for k, v in project.annotations.items()},
            "rounds": [{"name": r["name"], "model": r["model"], "frames": len(r["frame_ids"]), "status": r["status"]} for r in rounds],
            "models": [{"name": m["name"], "init": m["init"], "status": m["status"], "train_frames": m["counts"]["train"]} for m in models]}

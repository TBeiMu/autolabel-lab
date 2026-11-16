"""Read and save editable browser annotations, keeping archives and raw predictions fixed."""
from pathlib import Path

from .labels import make_document, parse_document
from .project import LabError, read_json, safe_name, sha256, write_json


def annotation_target(project, frame_id, subset=None, round_name=None):
    project.require_split()
    if bool(subset) == bool(round_name):
        raise LabError("请选择一个标注子集或审核轮次。")
    frame = next((f for f in project.frames() if f["id"] == frame_id), None)
    if not frame:
        raise LabError("图片不属于当前实验。")
    project.image_path(frame)
    raw = None
    record = None
    if round_name:
        round_name = safe_name(round_name)
        record = project.round(round_name)
        if record["status"] not in {"predicted", "reviewed"}:
            raise LabError("请等待本轮预测完成后再审核。")
        if frame_id not in record["frame_ids"] or frame["role"] != "pool":
            raise LabError("图片不属于所选审核轮次。")
        key = f"round_{round_name}"
        raw_path = project.root / "rounds" / round_name / "raw" / f"{frame_id}.json"
        if not raw_path.is_file() or sha256(raw_path) != record["raw_hashes"].get(frame_id):
            raise LabError("原始预测缺失或被修改。")
        raw = read_json(raw_path)
        folder = project.root / "rounds" / round_name / "editable"
    else:
        if subset not in {"seed", "val", "test"} or frame["role"] != subset:
            raise LabError("图片不属于所选人工标注子集。")
        key = subset
        folder = project.root / "workspaces" / subset
    archived = key in project.annotations
    if archived:
        folder = project.root / project.require_annotations(key)["folder"]
    elif record and record["status"] == "reviewed":
        raise LabError("已审核轮次缺少归档，不能编辑。")
    return frame, folder / f"{frame_id}.json", archived, raw, record


def read_annotation(project, frame_id, subset=None, round_name=None):
    frame, path, archived, raw, _ = annotation_target(project, frame_id, subset, round_name)
    document = read_json(path) if path.is_file() else make_document(frame, [])
    # Predictions can have out-of-bounds corners: display them so the user can repair them.
    parse_document(document, frame, project.config["classes"], check_bounds=False)
    return {"frame": frame, "document": document, "revision": sha256(path) if path.is_file() else None,
            "saved": path.is_file(), "read_only": archived, "raw": raw,
            "reviewed": bool(document.get("flags", {}).get("autolabel_lab_reviewed"))}


def save_annotation(project, frame_id, shapes, revision, subset=None, round_name=None):
    # Serialize with CLI imports and training snapshot creation; never edit a frozen archive.
    with project.lock():
        frame, path, archived, _, record = annotation_target(project, frame_id, subset, round_name)
        if archived:
            raise LabError("此标注已固定归档，只能查看；修改请建立新实验。")
        current = sha256(path) if path.is_file() else None
        if current != revision:
            raise LabError("标注已在另一页面修改，请重新加载后再保存；当前草稿已保留。")
        document = make_document(frame, [])
        if not isinstance(shapes, list) or len(shapes) > 5000:
            raise LabError("框列表必须是列表，且每张不超过 5000 个框。")
        document["shapes"] = shapes
        parsed = parse_document(document, frame, project.config["classes"])
        for shape in parsed:
            shape.pop("score", None)
        saved = make_document(frame, parsed)
        saved["flags"]["autolabel_lab_reviewed"] = True
        saved["flags"]["annotation_editor"] = "web"
        # Mark the round before writing the annotation; after an interrupted save, missing
        # per-image confirmations still prevent a whole-round import.
        if record and record.get("annotation_editor") != "web":
            record["annotation_editor"] = "web"
            write_json(project.root / "rounds" / round_name / "round.json", record)
        write_json(path, saved)
        return {"revision": sha256(path), "document": saved, "reviewed": True, "saved": True}

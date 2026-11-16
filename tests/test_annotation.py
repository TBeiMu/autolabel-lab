import cv2
import numpy as np
import pytest

from autolabel_lab.annotation import read_annotation, save_annotation
from autolabel_lab.dataset import import_labels
from autolabel_lab.labels import make_document
from autolabel_lab.project import Project, LabError, fingerprint, sha256, write_json


@pytest.fixture
def project(tmp_path):
    p = Project.create(tmp_path / "experiment", ["target", "other"])
    frames = []
    for i, role in enumerate(["seed", "seed", "val", "test", "pool", "pool"]):
        image = p.root / "frames" / f"f{i}.png"
        image.write_bytes(cv2.imencode(".png", np.full((100, 160, 3), 30 + i, np.uint8))[1].tobytes())
        frames.append({"id": f"f{i}", "file": image.name, "width": 160, "height": 100,
                       "role": role, "image_sha256": sha256(image), "frame_index": i, "timestamp": i})
    write_json(p.root / "manifest.json", {"frames": frames, "split": {"frames_hash": fingerprint(frames)}})
    raw_hashes = {}
    for f in p.frames("pool"):
        doc = make_document(f, [{"label": "target", "points": [[20, 20], [60, 20], [60, 50], [20, 50]], "score": .8}], prediction=True)
        raw = p.root / f"rounds/r1/raw/{f['id']}.json"
        write_json(raw, doc)
        write_json(p.root / f"rounds/r1/editable/{f['id']}.json", doc)
        raw_hashes[f["id"]] = sha256(raw)
    write_json(p.root / "rounds/r1/round.json", {"name": "r1", "status": "predicted", "frame_ids": ["f4", "f5"], "raw_hashes": raw_hashes})
    return p


def shape(label="target"):
    return {"label": label, "points": [[20, 20], [60, 20], [60, 50], [20, 50]], "shape_type": "rotation", "flags": {}}


def test_save_reload_empty_negative_and_conflicting_browser_revision(project):
    data = read_annotation(project, "f0", subset="seed")
    assert data["saved"] is False and data["revision"] is None
    saved = save_annotation(project, "f0", [shape()], None, subset="seed")
    assert read_annotation(project, "f0", subset="seed")["document"]["shapes"][0]["label"] == "target"
    with pytest.raises(LabError, match="另一页面"):
        save_annotation(project, "f0", [], None, subset="seed")
    assert len(read_annotation(project, "f0", subset="seed")["document"]["shapes"]) == 1
    save_annotation(project, "f0", [], saved["revision"], subset="seed")
    assert read_annotation(project, "f0", subset="seed")["saved"]
    assert read_annotation(project, "f0", subset="seed")["document"]["shapes"] == []


def test_save_rejects_wrong_subset_unknown_classes_and_invalid_geometry(project):
    with pytest.raises(LabError, match="子集"):
        read_annotation(project, "f2", subset="seed")
    with pytest.raises(LabError, match="轮次"):
        read_annotation(project, "f2", round_name="r1")
    for invalid in [shape("unknown"), {**shape(), "points": [[-1, 20], [60, 20], [60, 50], [-1, 50]]},
                    {**shape(), "points": [[0, 0]] * 4}]:
        with pytest.raises(LabError):
            save_annotation(project, "f0", [invalid], None, subset="seed")
    assert not (project.root / "workspaces/seed/f0.json").exists()
    assert not (project.root / ".operation.lock").exists()


def test_archived_annotations_stay_read_only(project):
    for fid in ["f0", "f1"]:
        save_annotation(project, fid, [shape()], None, subset="seed")
    import_labels(project, subset="seed")
    archived = project.root / "annotations/seed/f0.json"
    before = sha256(archived)
    data = read_annotation(project, "f0", subset="seed")
    assert data["read_only"]
    with pytest.raises(LabError, match="归档"):
        save_annotation(project, "f0", [], data["revision"], subset="seed")
    assert sha256(archived) == before


def test_browser_round_requires_each_frame_saved_and_preserves_raw_predictions(project):
    raw_before = {fid: sha256(project.root / f"rounds/r1/raw/{fid}.json") for fid in ["f4", "f5"]}
    data = read_annotation(project, "f4", round_name="r1")
    assert data["raw"]["shapes"][0]["score"] == .8
    save_annotation(project, "f4", [shape("other")], data["revision"], round_name="r1")
    with pytest.raises(LabError, match="尚未在网页"):
        import_labels(project, round_name="r1", reviewed=True)
    assert "round_r1" not in project.annotations
    data = read_annotation(project, "f5", round_name="r1")
    save_annotation(project, "f5", [], data["revision"], round_name="r1")
    import_labels(project, round_name="r1", reviewed=True)
    assert read_annotation(project, "f4", round_name="r1")["read_only"]
    assert project.require_annotations("round_r1")["boxes"] == 1
    assert {fid: sha256(project.root / f"rounds/r1/raw/{fid}.json") for fid in raw_before} == raw_before


def test_operation_lock_blocks_save_without_losing_disk_annotation(project):
    (project.root / ".operation.lock").write_text("active operation")
    with pytest.raises(LabError, match="另一个操作"):
        save_annotation(project, "f0", [shape()], None, subset="seed")
    assert not (project.root / "workspaces/seed/f0.json").exists()


def test_image_integrity_is_checked_before_editing(project):
    (project.root / "frames/f0.png").write_bytes(b"changed")
    with pytest.raises(LabError, match="图片"):
        read_annotation(project, "f0", subset="seed")


def test_annotation_http_requires_token_valid_revision_and_blocks_background_task(project):
    import json
    import threading
    import urllib.request
    import urllib.error
    from autolabel_lab.web import make_server
    server = make_server(project.root.parent, port=0)
    entry = server.workspace.register(project.root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    payload = {"project_id": entry["id"], "frame": "f0", "subset": "seed", "revision": None, "shapes": [shape()]}
    def post(body, token=None):
        headers = {"Content-Type": "application/json"}
        if token: headers["X-Lab-Token"] = token
        req = urllib.request.Request(url + "/api/annotation", data=json.dumps(body).encode(), headers=headers)
        return urllib.request.urlopen(req, timeout=5)
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            post(payload)
        assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as error:
            post({k: v for k, v in payload.items() if k != "revision"}, server.workspace.token)
        assert error.value.code == 400
        # A queued worker for this project blocks edits before the CLI lock exists.
        server.workspace.jobs["pending"] = {"id": "pending", "status": "queued", "project_id": entry["id"],
            "created_at": "2099-01-01T00:00:00+00:00", "worker_token": "future"}
        with pytest.raises(urllib.error.HTTPError) as error:
            post(payload, server.workspace.token)
        assert "后台任务" in json.loads(error.value.read())["error"]
        server.workspace.jobs.clear()
        with post(payload, server.workspace.token) as response:
            saved = json.load(response)
        with urllib.request.urlopen(url + f"/api/annotation?id={entry['id']}&frame=f0&subset=seed") as response:
            loaded = json.load(response)
        assert loaded["revision"] == saved["revision"]
        assert loaded["document"]["shapes"][0]["label"] == "target"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

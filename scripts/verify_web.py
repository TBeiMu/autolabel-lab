"""Verify the web task queue using synthetic data; never use as experimental evidence."""
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from smoke import points
from autolabel_lab.labels import make_document
from autolabel_lab.project import Project, read_json, write_json


def run(url, root, training=False, device="cpu"):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Choose a new output directory; existing data is never overwritten.")
    root.mkdir(parents=True)
    video = root / "synthetic.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 160))
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic video")
    for index in range(120):
        image = np.full((160, 160, 3), 230, np.uint8)
        cv2.fillConvexPoly(image, np.asarray(points(index), np.int32), (10, 30, 200))
        cv2.putText(image, str(index), (5, 150), cv2.FONT_HERSHEY_SIMPLEX, .5, (10, 10, 10), 1)
        writer.write(image)
    writer.release()
    project = Project.create(root / "experiment", ["synthetic_target"])
    project.config["synthetic_smoke_test"] = True
    write_json(project.root / "project.json", project.config)
    token = ""

    def api(path, payload=None):
        req = urllib.request.Request(url.rstrip("/") + path,
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"Content-Type": "application/json", "X-Lab-Token": token})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read())

    token = api("/api/bootstrap")["token"]
    entry = api("/api/register", {"path": str(project.root)})
    verified = []

    def job(action, payload):
        created = api("/api/job", {"project_id": entry["id"], "action": action, "payload": payload})
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            current = next(j for j in api("/api/overview")["jobs"] if j["id"] == created["id"])
            if current["status"] not in {"queued", "running", "cancelling"}:
                if current["status"] != "completed":
                    raise RuntimeError(api("/api/log?id=" + created["id"])["text"])
                assert not (project.root / ".operation.lock").exists()
                verified.append({"action": action, "job": created["id"], "payload": payload})
                print(f"Passed web job: {action} {payload.get('name', payload.get('round', ''))}", flush=True)
                return
            time.sleep(.3)
        raise TimeoutError(f"Job did not finish: {created['id']}")

    preview = api("/api/preview-video", {"path": str(video), "second": 1})
    assert preview["width"] == 160 and preview["image"].startswith("data:image/jpeg;base64,")
    job("extract", {"video": str(video), "every": .2, "max_frames": 100})
    job("split", {"seed_count": 4, "gap_seconds": .2})
    for subset in ["seed", "val", "test"]:
        for frame in project.frames(subset):
            write_json(project.root / "workspaces" / subset / f"{frame['id']}.json",
                       make_document(frame, [{"label": "synthetic_target", "points": points(frame["frame_index"])}]))
        job("import-labels", {"subset": subset})
    params = {"name": "M0", "epochs": 2, "imgsz": 64, "batch": 2, "device": device}
    job("train", {**params, "dry_run": True})
    if training:
        job("train", params)
        job("autolabel", {"round": "r1", "model": "M0", "count": 6, "conf": .05, "device": device})
        for frame_id in project.round("r1")["frame_ids"]:
            frame = next(f for f in project.frames() if f["id"] == frame_id)
            write_json(project.root / "rounds" / "r1" / "editable" / f"{frame_id}.json",
                       make_document(frame, [{"label": "synthetic_target", "points": points(frame["frame_index"])}]))
        job("import-labels", {"round": "r1", "reviewed": True})
        job("train", {**params, "name": "M1", "rounds": ["r1"]})
        job("evaluate", {"models": ["M0", "M1"], "device": device})
        details = api("/api/project?id=" + entry["id"])
        assert all(m["model_yaml"] == "yolo26s-obb.yaml" and len(m["curve"]) == 2 for m in details["models"])
        assert len({m["evaluation"]["test_hash"] for m in details["models"]}) == 1

        # Exercise cancellation inside a real Trainer callback, then verify lock release.
        cancelled = api("/api/job", {"project_id": entry["id"], "action": "train",
                                    "payload": {**params, "name": "cancel_check", "epochs": 100}})
        record = project.root / "models" / "cancel_check" / "model.json"
        deadline = time.monotonic() + 60
        while not record.exists() and time.monotonic() < deadline:
            time.sleep(.1)
        assert record.exists(), "Cancellation test did not reach Trainer setup"
        api("/api/cancel", {"id": cancelled["id"]})
        while time.monotonic() < deadline:
            current = next(j for j in api("/api/overview")["jobs"] if j["id"] == cancelled["id"])
            if current["status"] == "cancelled":
                break
            time.sleep(.2)
        assert current["status"] == "cancelled"
        assert "停止" in read_json(record)["error"]
        assert not (project.root / ".operation.lock").exists()
        print("Passed cooperative cancellation and project lock release", flush=True)
    job("report", {})
    details = api("/api/project?id=" + entry["id"])
    assert details["report"]["synthetic_smoke_test"]
    result = {"synthetic_smoke_test": True, "training_executed": training, "project_id": entry["id"],
              "project": str(project.root), "verified_jobs": verified, "cancellation_verified": training}
    write_json(root / "web_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", default=f"_smoke/web_{uuid.uuid4().hex[:8]}")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.url, args.out, args.training, args.device)

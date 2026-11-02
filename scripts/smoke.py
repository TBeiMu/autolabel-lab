"""Synthetic integration check. The results are not a real annotation experiment."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from autolabel_lab.dataset import extract, import_labels, split
from autolabel_lab.labels import make_document
from autolabel_lab.models import autolabel, evaluate, train
from autolabel_lab.project import Project, read_json, write_json
from autolabel_lab.reports import report


def points(index):
    angle = (index * 2) % 90
    center = (75 + 20 * math.sin(index / 10), 80 + 15 * math.cos(index / 8))
    return cv2.boxPoints((center, (44, 24), angle)).astype(float).tolist()


def run(root, training=False, device="cpu"):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError(f"Smoke directory already exists; choose a new path: {root}")
    root.mkdir(parents=True)
    video = root / "synthetic.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 160))
    if not writer.isOpened():
        raise RuntimeError("Could not create the synthetic video")
    for i in range(120):
        image = np.full((160, 160, 3), 230, np.uint8)
        cv2.fillConvexPoly(image, np.asarray(points(i), np.int32), (10, 30, 200))
        cv2.putText(image, str(i), (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
        writer.write(image)
    writer.release()
    project = Project.create(root / "experiment", ["synthetic_target"])
    project.config["synthetic_smoke_test"] = True
    write_json(project.root / "project.json", project.config)
    extract(project, video, every=0.2)
    split(project, seed_count=4, gap_seconds=0.2)
    for subset in ["seed", "val", "test"]:
        for frame in project.frames(subset):
            doc = make_document(frame, [{"label": "synthetic_target", "points": points(frame["frame_index"])}])
            write_json(project.root / "workspaces" / subset / f"{frame['id']}.json", doc)
        import_labels(project, subset=subset)
    train(project, "M0", epochs=1, imgsz=64, batch=2, device=device, dry_run=not training)
    if training:
        autolabel(project, "r1", "M0", count=6, conf=0.05, device=device)
        # Use generated geometry as the synthetic human oracle; no fake work time.
        for frame_id in project.round("r1")["frame_ids"]:
            frame = next(f for f in project.frames() if f["id"] == frame_id)
            write_json(project.root / "rounds" / "r1" / "editable" / f"{frame_id}.json",
                       make_document(frame, [{"label": "synthetic_target", "points": points(frame["frame_index"])}]))
        import_labels(project, round_name="r1", reviewed=True)
        train(project, "M1", rounds=["r1"], epochs=1, imgsz=64, batch=2, device=device)
        evaluate(project, ["M0", "M1"], device=device)
    result = report(project)
    write_json(root / "smoke_result.json", {"synthetic_smoke_test": True, "training_executed": training, "artifacts": result})
    print(json.dumps({"synthetic_smoke_test": True, "training_executed": training, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="_smoke/core")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.out, args.training, args.device)

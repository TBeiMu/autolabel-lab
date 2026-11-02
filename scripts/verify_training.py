"""Run the synthetic GPU workflow and verify checkpoint transfer with the real trainer."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import OBBTrainer

from smoke import run
from autolabel_lab.models import evaluate, train
from autolabel_lab.project import Project, write_json
from autolabel_lab.reports import report


def verify(out, device):
    builds = []
    get_model = OBBTrainer.get_model

    def track_build(trainer, cfg=None, weights=None, verbose=True):
        builds.append(weights is not None)
        return get_model(trainer, cfg=cfg, weights=weights, verbose=verbose)

    with patch.object(OBBTrainer, "get_model", track_build):
        run(out, training=True, device=device)
    if builds != [False, False]:
        raise AssertionError(f"Scratch models received checkpoint weights: {builds}")

    project = Project(Path(out) / "experiment")
    parent_record, parent_path = project.model("M0")
    parent = YOLO(str(parent_path), task="obb").model.state_dict()
    checked_tensors = []

    def check_transfer(trainer, cfg=None, weights=None, verbose=True):
        if weights is None:
            raise AssertionError("Continue training discarded the parent checkpoint")
        model = get_model(trainer, cfg=cfg, weights=weights, verbose=verbose)
        transferred = model.state_dict()
        if transferred.keys() != parent.keys():
            raise AssertionError("Continue model state keys differ from the parent")
        for key, expected in parent.items():
            if not torch.equal(transferred[key].cpu().float(), expected.cpu().float()):
                raise AssertionError(f"Parent tensor was not transferred: {key}")
            checked_tensors.append(key)
        return model

    with patch.object(OBBTrainer, "get_model", check_transfer):
        continued = train(project, "M1_continue", rounds=["r1"], init="continue", from_model="M0",
                          epochs=1, imgsz=64, batch=2, device=device)
    if not checked_tensors or continued["parent_weights_sha256"] != parent_record["weights_sha256"]:
        raise AssertionError("Continue training provenance check failed")
    evaluate(project, ["M1_continue"], device=device)
    artifacts = report(project)
    result = {"synthetic_smoke_test": True, "scratch_builds_without_weights": len(builds),
              "parent_tensors_verified_before_training": len(checked_tensors),
              "continue_completed": continued["status"] == "completed", "artifacts": artifacts}
    write_json(Path(out) / "training_verification.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New directory for synthetic verification artifacts")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    verify(args.out, args.device)

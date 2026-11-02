from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .dataset import extract, import_labels, split
from .models import autolabel, evaluate, train
from .project import DEFAULT_MODEL_YAML, LabError, Project
from .reports import record_time, report, status


def comma_list(value):
    result = [v.strip() for v in value.split(",")]
    if not all(result):
        raise argparse.ArgumentTypeError("逗号分隔列表不能包含空项。")
    return result


def timestamp_list(value):
    try:
        return [float(v) for v in comma_list(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("时间点必须是秒数，例如 0,2.5,10。") from exc


def parser():
    p = argparse.ArgumentParser(description="视频抽帧 → X-AnyLabeling 人工标注 → 从零训练 → 自动标注 → 修正与对比")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--project", required=True, help="实验数据目录，例如 experiments/demo")
    sub = p.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("init", help="创建 OBB 实验")
    cmd.add_argument("--classes", required=True, type=comma_list, help="固定类别顺序，例如 object_a,object_b")
    cmd.add_argument("--seed", type=int, default=42)
    cmd = sub.add_parser("extract", help="按间隔或指定时间点抽帧（恒定帧率视频）")
    cmd.add_argument("--video", required=True)
    group = cmd.add_mutually_exclusive_group()
    group.add_argument("--every", type=float, default=1.0, help="抽帧间隔秒数，默认 1")
    group.add_argument("--timestamps", type=timestamp_list, help="指定秒数，例如 0,2.5,10")
    cmd.add_argument("--start", type=float, default=0.0)
    cmd.add_argument("--end", type=float)
    cmd.add_argument("--max-frames", type=int, default=1000)
    cmd = sub.add_parser("split", help="按连续时间段固定划分并生成标注工作目录")
    cmd.add_argument("--seed-count", type=int, default=15)
    cmd.add_argument("--val-fraction", type=float, default=0.15)
    cmd.add_argument("--test-fraction", type=float, default=0.2)
    cmd.add_argument("--gap-seconds", type=float, default=2.0)
    cmd = sub.add_parser("import-labels", help="验证并归档 X-AnyLabeling JSON；整批完成后导入")
    target = cmd.add_mutually_exclusive_group(required=True)
    target.add_argument("--subset", choices=["seed", "val", "test"])
    target.add_argument("--round", dest="round_name")
    cmd.add_argument("--from", dest="source", help="JSON 所在目录，默认使用本实验工作目录")
    cmd.add_argument("--reviewed", action="store_true", help="确认已逐张人工审核（含无预测帧）")
    cmd = sub.add_parser("train", help="训练并固定模型及数据快照")
    cmd.add_argument("--name", required=True)
    cmd.add_argument("--rounds", type=comma_list, default=[], help="纳入已审核轮次，例如 r1,r2；默认仅种子集")
    cmd.add_argument("--init", choices=["scratch", "continue"], default="scratch")
    cmd.add_argument("--from-model")
    cmd.add_argument("--model-yaml", help=f"OBB 结构 YAML，新实验默认 {DEFAULT_MODEL_YAML}；已有实验使用其 project.json 配置")
    cmd.add_argument("--epochs", type=int, default=100)
    cmd.add_argument("--imgsz", type=int, default=640)
    cmd.add_argument("--batch", type=int, default=4)
    cmd.add_argument("--device", default="cpu", help="cpu 或 GPU 编号 0")
    cmd.add_argument("--dry-run", action="store_true", help="仅导出数据快照和训练计划，不加载训练库")
    cmd = sub.add_parser("autolabel", help="用已训练模型预标注未使用的训练池图片")
    cmd.add_argument("--round", dest="name", required=True)
    cmd.add_argument("--model", dest="model_name", required=True)
    group = cmd.add_mutually_exclusive_group()
    group.add_argument("--count", type=int, default=25)
    group.add_argument("--frames", dest="frame_ids", type=comma_list, help="指定 manifest 中的图片 id")
    cmd.add_argument("--conf", type=float, default=0.25)
    cmd.add_argument("--iou", type=float, default=0.7)
    cmd.add_argument("--imgsz", type=int)
    cmd.add_argument("--device", default="cpu")
    cmd = sub.add_parser("evaluate", help="在固定人工测试集评价模型")
    cmd.add_argument("--models", dest="names", type=comma_list, required=True)
    cmd.add_argument("--device", default="cpu")
    cmd = sub.add_parser("time", help="记录实测人工作时；多次执行表示不同工作时段累加")
    cmd.add_argument("--stage", choices=["seed", "manual", "review"], required=True)
    cmd.add_argument("--seconds", type=float, required=True)
    cmd.add_argument("--round", dest="round_name")
    cmd.add_argument("--frame", dest="frame_id", help="单图工时；不填表示整批完成该时段")
    cmd.add_argument("--note", default="")
    cmd = sub.add_parser("report", help="输出 Markdown、JSON 和逐图 CSV 实验报告")
    cmd.add_argument("--match-iou", type=float, default=0.5)
    cmd.add_argument("--unchanged-iou", type=float, default=0.95)
    sub.add_parser("status", help="查看数据与实验状态")
    return p


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command, root = args.pop("command"), args.pop("project")
    try:
        if command == "init":
            project = Project.create(root, **args)
            result = status(project)
        else:
            project = Project(root)
            actions = {"extract": extract, "split": split, "import-labels": import_labels, "train": train,
                       "autolabel": autolabel, "evaluate": evaluate, "time": record_time, "report": report, "status": status}
            if command == "status":
                result = status(project)
            else:
                with project.lock():
                    result = actions[command](project, **args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (LabError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

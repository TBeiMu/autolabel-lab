import os
from pathlib import Path

from .project import LabError


def check_cancelled(*_):
    marker = os.environ.get("ALAB_CANCEL_FILE")
    if marker and Path(marker).is_file():
        raise LabError("任务已由用户停止。")

"""Durable CLI worker; its lifetime and completion do not depend on the web server."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import psutil

from .project import now, read_json, write_json


def main():
    folder = Path(sys.argv[1]).resolve()
    job = read_json(folder / "job.json")
    process = psutil.Process()
    write_json(folder / "worker.json", {"pid": process.pid, "create_time": process.create_time(),
                                       "token": job["worker_token"]})
    code = 0
    try:
        from .cancellation import check_cancelled
        check_cancelled()
        from .cli import main as cli_main
        cli_main(sys.argv[2:])
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1 if exc.code else 0
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        write_json(folder / "result.json", {"exit_code": code, "completed_at": now(),
                                          "token": job["worker_token"]})
    return code


if __name__ == "__main__":
    raise SystemExit(main())

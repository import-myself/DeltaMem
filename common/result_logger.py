"""
Result Logger — CSV-based run deduplication & result tracking.

Usage (Lock 1 check + final write):
    from common.result_logger import check_existing_result, append_result

    key = {"traj_dir": os.path.abspath(traj_dir)}
    if check_existing_result(csv_path, key):
        logger.info("Already done, skipping.")
        return

    # ... run ...

    append_result(csv_path, {"traj_dir": ..., "success_rate": ..., ...})
"""

import csv
import fcntl
import os
import time
from typing import Any, Dict, Optional


def check_existing_result(csv_path: str, key_params: Dict[str, Any]) -> Optional[Dict]:
    """
    Return the first CSV row where every key in key_params matches, or None.
    Treats all values as strings for comparison.
    """
    if not os.path.exists(csv_path):
        return None
    target = {k: str(v) for k, v in key_params.items()}
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if all(row.get(k) == target[k] for k in target):
                    return dict(row)
    except Exception:
        return None
    return None


def append_result(csv_path: str, row: Dict[str, Any]) -> None:
    """
    Append one result row to csv_path, creating the file with a header if needed.
    Uses an exclusive file lock so concurrent processes don't corrupt the file.
    """
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(csv_dir, exist_ok=True)

    row_str = {k: str(v) for k, v in row.items()}
    fieldnames = list(row_str.keys())

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)                    # seek to end
            needs_header = f.tell() == 0    # empty file → write header
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow(row_str)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

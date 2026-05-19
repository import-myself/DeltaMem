"""
Memory path detection utilities.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MEMORY_LABELS = {
    "prtree":        "DeltaMem (Dual PR-Tree)",
    "synapse":       "Synapse",
    "awm":           "AWM",
    "reasoningbank": "ReasoningBank",
    "no-memory":     "No Memory",
}


def detect_memory_type(path: str) -> Optional[str]:
    """Infer memory type from files present at path. Returns method name or None if empty/unknown."""
    p = Path(path)

    # prtree: path is a prefix; actual files are {path}_task.json / {path}_env.json
    if Path(f"{path}_task.json").exists() or Path(f"{path}_env.json").exists():
        return "prtree"

    if not p.is_dir():
        return None

    # synapse: directory containing exemplars.json or specifiers.json
    if (p / "exemplars.json").exists() or (p / "specifiers.json").exists():
        return "synapse"

    # reasoningbank: directory containing {benchmark}/memory_pool.json
    for subdir in p.iterdir():
        if subdir.is_dir() and (subdir / "memory_pool.json").exists():
            return "reasoningbank"

    # awm: directory containing {benchmark}/*.txt files
    for subdir in p.iterdir():
        if subdir.is_dir() and list(subdir.glob("*.txt")):
            return "awm"

    return None


def check_memory_path(memory_path: str, expected_method: str) -> bool:
    """
    Validate that the files at memory_path match expected_method.
    Logs a warning if mismatch detected; logs info if path is empty (cold start).
    Returns True if safe to proceed, False if mismatch.
    """
    detected = detect_memory_type(memory_path)

    if detected is None:
        logger.info(f"🌱 [{expected_method}] No existing memory at '{memory_path}', cold start.")
        return True

    if detected != expected_method:
        detected_label  = _MEMORY_LABELS.get(detected,       detected)
        expected_label  = _MEMORY_LABELS.get(expected_method, expected_method)
        logger.warning(
            f"⚠️  Memory path '{memory_path}' contains {detected_label} data, "
            f"but current method is {expected_label}. "
            f"Please verify --memory-path is correct. Aborting load."
        )
        return False

    label = _MEMORY_LABELS.get(expected_method, expected_method)
    logger.info(f"📥 [{label}] Found existing memory at '{memory_path}', will load.")
    return True

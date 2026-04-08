# core/training_metadata.py
#
# Training metadata tracking, auto-retrain logic, locking, and model naming.
#
# Artifact: artifacts/training_metadata.json
# Schema:
#   retrains[]           — log of all retrain events
#   current_threshold    — new-corrections threshold for next retrain
#   corrections_at_last_retrain — feedback row count at time of last retrain
#   lock                 — None or {lock_id, timestamp} if retrain in progress
#   next_version_number  — sequential counter for ac_vNNN naming
#   settings             — base_threshold, min/max, growth_factor, stale_lock_hours

import os
import math
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

from .config import CONFIG
from .file_utils import atomic_write
from .feedback import load_feedback

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────

_DEFAULTS = {
    "retrains": [],
    "current_threshold": 200,
    "corrections_at_last_retrain": 0,
    "lock": None,
    "next_version_number": 1,
    "settings": {
        "base_threshold": 200,
        "min_threshold": 50,
        "max_threshold": 1000,
        "growth_factor": 0.1,
        "stale_lock_hours": 2,
    },
}


def _meta_path() -> str:
    return CONFIG.get("training_metadata_json") or os.path.join(
        CONFIG["artifacts_dir"], "training_metadata.json"
    )


def get_metadata() -> dict:
    """Load training_metadata.json, creating with defaults if absent."""
    path = _meta_path()
    if not os.path.exists(path):
        return init_metadata()
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Back-fill any missing keys from defaults
        for k, v in _DEFAULTS.items():
            if k not in data:
                data[k] = v
        return data
    except Exception:
        return init_metadata()


def save_metadata(data: dict) -> None:
    atomic_write(_meta_path(), data)


def init_metadata() -> dict:
    """Create training_metadata.json with default values."""
    import copy
    data = copy.deepcopy(_DEFAULTS)
    # Merge user-configured settings from CONFIG if present
    cfg_settings = CONFIG.get("retrain_settings", {})
    if cfg_settings:
        data["settings"].update(cfg_settings)
        data["current_threshold"] = cfg_settings.get(
            "base_threshold", data["current_threshold"]
        )
    save_metadata(data)
    return data


# ─────────────────────────────────────────────────────────────
# Version naming: ac_vNNN
# ─────────────────────────────────────────────────────────────

def next_version_name() -> str:
    """
    Generate the next sequential model version name: ac_v001, ac_v002, etc.
    Increments next_version_number in metadata.
    """
    meta = get_metadata()
    num = meta.get("next_version_number", 1)
    name = f"ac_v{num:03d}"
    meta["next_version_number"] = num + 1
    save_metadata(meta)
    return name


def peek_next_version_name() -> str:
    """Preview the next version name without incrementing."""
    meta = get_metadata()
    num = meta.get("next_version_number", 1)
    return f"ac_v{num:03d}"


# ─────────────────────────────────────────────────────────────
# Corrections tracking
# ─────────────────────────────────────────────────────────────

def get_corrections_since_last_retrain() -> int:
    """
    Count how many feedback rows have accumulated since the last retrain.
    Uses corrections_at_last_retrain as the watermark.
    """
    meta = get_metadata()
    watermark = meta.get("corrections_at_last_retrain", 0)
    fb = load_feedback()
    current_count = len(fb)
    return max(0, current_count - watermark)


def get_retrain_status() -> Dict[str, Any]:
    """
    Return a summary of the current retrain status:
      - new_corrections: corrections since last retrain
      - threshold: current threshold
      - should_retrain: True if corrections >= threshold
      - is_locked: True if a retrain is in progress
      - progress_pct: 0-100 (corrections / threshold as percentage)
      - total_feedback: total feedback rows
      - retrains_completed: number of past retrains
    """
    meta = get_metadata()
    new_corrections = get_corrections_since_last_retrain()
    threshold = meta.get("current_threshold", 200)
    locked, _ = is_locked()
    fb = load_feedback()

    return {
        "new_corrections": new_corrections,
        "threshold": threshold,
        "should_retrain": new_corrections >= threshold and not locked,
        "is_locked": locked,
        "progress_pct": min(100, int(100 * new_corrections / max(threshold, 1))),
        "total_feedback": len(fb),
        "retrains_completed": len(meta.get("retrains", [])),
        "next_version": peek_next_version_name(),
        "last_retrain_at": meta["retrains"][-1]["timestamp"] if meta.get("retrains") else None,
    }


# ─────────────────────────────────────────────────────────────
# Threshold logic
# ─────────────────────────────────────────────────────────────

def compute_next_threshold(dataset_size: int, num_retrains: int) -> int:
    """
    Dynamic threshold: as the dataset grows and more retrains happen,
    require more corrections to meaningfully change the model.

    Formula:
        threshold = base + growth_factor * dataset_size * log2(num_retrains + 2)
        clamped to [min_threshold, max_threshold]

    Logic: Early retrains are cheap (small dataset, few retrains).
    Later retrains need more corrections because the marginal value
    of each new sample decreases.
    """
    meta = get_metadata()
    settings = meta.get("settings", _DEFAULTS["settings"])

    base = settings.get("base_threshold", 200)
    factor = settings.get("growth_factor", 0.1)
    floor = settings.get("min_threshold", 50)
    ceiling = settings.get("max_threshold", 1000)

    raw = base + factor * math.sqrt(dataset_size) * math.log2(num_retrains + 2)
    return max(floor, min(ceiling, int(round(raw))))


# ─────────────────────────────────────────────────────────────
# Locking — only one retrain at a time
# ─────────────────────────────────────────────────────────────

def acquire_lock() -> Tuple[bool, Optional[str]]:
    """
    Try to acquire the retrain lock.
    Returns (success, lock_id). If already locked (and not stale),
    returns (False, None).
    """
    meta = get_metadata()
    locked, existing_lock_id = is_locked()

    if locked:
        logger.warning("Retrain lock already held: %s", existing_lock_id)
        return False, None

    lock_id = uuid.uuid4().hex[:12]
    meta["lock"] = {
        "lock_id": lock_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_metadata(meta)
    logger.info("Acquired retrain lock: %s", lock_id)
    return True, lock_id


def release_lock(lock_id: Optional[str] = None) -> bool:
    """
    Release the retrain lock. If lock_id is provided, only release
    if it matches (prevents accidental release by another process).
    """
    meta = get_metadata()
    current = meta.get("lock")

    if current is None:
        return True  # already unlocked

    if lock_id and current.get("lock_id") != lock_id:
        logger.warning(
            "Lock release denied: expected %s, found %s",
            lock_id, current.get("lock_id"),
        )
        return False

    meta["lock"] = None
    save_metadata(meta)
    logger.info("Released retrain lock: %s", current.get("lock_id"))
    return True


def is_locked() -> Tuple[bool, Optional[str]]:
    """
    Check if a retrain is currently in progress.
    Stale locks (older than stale_lock_hours) are auto-released.
    Returns (is_locked, lock_id_or_None).
    """
    meta = get_metadata()
    lock = meta.get("lock")

    if lock is None:
        return False, None

    # Check for stale lock
    stale_hours = meta.get("settings", {}).get("stale_lock_hours", 2)
    try:
        lock_time = datetime.fromisoformat(lock["timestamp"])
        if datetime.now(timezone.utc) - lock_time > timedelta(hours=stale_hours):
            logger.warning("Stale lock detected (held since %s), auto-releasing.", lock["timestamp"])
            meta["lock"] = None
            save_metadata(meta)
            return False, None
    except Exception:
        pass

    return True, lock.get("lock_id")


# ─────────────────────────────────────────────────────────────
# Retrain event logging
# ─────────────────────────────────────────────────────────────

def log_retrain_event(
    version: str,
    dataset_size: int,
    new_corrections: int,
    accuracy: Optional[float] = None,
    f1_macro: Optional[float] = None,
    promoted: bool = False,
    previous_version: Optional[str] = None,
    training_run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    notes: str = "",
) -> dict:
    """
    Log a completed retrain event and update thresholds.
    Called after a successful retraining run.
    """
    meta = get_metadata()
    fb = load_feedback()

    retrain_id = len(meta.get("retrains", [])) + 1
    event = {
        "retrain_id": retrain_id,
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": dataset_size,
        "new_corrections": new_corrections,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "promoted": promoted,
        "previous_version": previous_version,
        "threshold_used": meta.get("current_threshold", 200),
        "training_run_id": training_run_id,
        "parent_run_id": parent_run_id,
        "duration_seconds": duration_seconds,
        "notes": notes,
    }

    meta.setdefault("retrains", []).append(event)

    # Update watermark: current feedback count becomes the new baseline
    meta["corrections_at_last_retrain"] = len(fb)

    # Compute next dynamic threshold
    num_retrains = len(meta["retrains"])
    meta["current_threshold"] = compute_next_threshold(dataset_size, num_retrains)

    save_metadata(meta)
    logger.info(
        "Retrain event logged: %s (acc=%.3f, threshold→%d)",
        version,
        accuracy or 0,
        meta["current_threshold"],
    )
    return event


# ─────────────────────────────────────────────────────────────
# Auto-compare: decide if new model should be promoted
# ─────────────────────────────────────────────────────────────

def should_auto_promote(
    new_accuracy: float,
    current_accuracy: Optional[float],
    new_f1: Optional[float] = None,
    current_f1: Optional[float] = None,
    min_improvement: float = 0.0,
) -> Tuple[bool, str]:
    """
    Decide whether a new model should be auto-promoted over the current one.

    Rules:
      1. If no current model exists → promote (first model)
      2. If new accuracy >= current accuracy + min_improvement → promote
      3. If accuracy is tied, compare F1 macro as tiebreaker
      4. Otherwise → keep current (new model stays as candidate)

    Returns (should_promote, reason).
    """
    if current_accuracy is None:
        return True, "First model — auto-promoted."

    acc_delta = new_accuracy - current_accuracy

    if acc_delta > min_improvement:
        return True, (
            f"Accuracy improved: {current_accuracy:.3f} → {new_accuracy:.3f} "
            f"(+{acc_delta:.3f})"
        )

    if abs(acc_delta) <= min_improvement and new_f1 is not None and current_f1 is not None:
        if new_f1 > current_f1:
            return True, (
                f"Accuracy tied ({new_accuracy:.3f}), "
                f"F1 improved: {current_f1:.3f} → {new_f1:.3f}"
            )

    return False, (
        f"No improvement: current {current_accuracy:.3f} vs new {new_accuracy:.3f} "
        f"(delta {acc_delta:+.3f}). New model kept as candidate only."
    )

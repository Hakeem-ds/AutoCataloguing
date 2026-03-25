import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from .config import CONFIG


# ============================================================
# Normalisation utilities
# ============================================================

def normalize_sys_id(value: Any) -> str:
    """
    Format-level normalisation:
       - convert to string
       - trim
       - uppercase
       - replace spaces/underscores -> hyphens
       - collapse multiple hyphens
    Preserves leading zeros and all alphanumerics.
    """
    if value is None:
        return ""
    s = str(value).strip().upper()
    if not s:
        return ""
    s = s.replace("_", "-")
    s = "-".join(part for part in s.split() if part)
    while "--" in s:
        s = s.replace("--", "-")
    return s


def resolve_hierarchical_sys_id(raw_sys_id: str, valid_folders: set[str]) -> Optional[str]:
    """
    Taxonomy-level normalisation:
    Given a SYSID like "A/B/C/D/E/F", progressively try:
        A/B/C/D/E/F
        A/B/C/D/E
        A/B/C/D
        A/B/C
        A/B
        A
    Return the first match in `valid_folders`, or None.
    """
    if not isinstance(raw_sys_id, str):
        return None

    parts = raw_sys_id.split("/")

    for i in range(len(parts), 0, -1):
        candidate = "/".join(parts[:i])
        if candidate in valid_folders:
            return candidate

    return None


# ============================================================
# Label map schema (sys_id-centric with aliases)
# ============================================================

def _empty_label_map() -> Dict[str, Any]:
    return {
        "seen_sys_ids": [],       # canonical sys_ids observed in training or feedback
        "aliases": {},            # canonical sys_id -> list of alias strings
        "version": datetime.utcnow().isoformat(),
        "version_note": None,
    }


def _atomic_write(path: str, data: dict):
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp, path)


def load_label_map() -> dict:
    """Load the sys_id/alias map. Auto-create if missing."""
    path = CONFIG["label_map_json"]

    if not os.path.exists(path):
        lm = _empty_label_map()
        _atomic_write(path, lm)
        return lm

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "seen_sys_ids" not in data or "aliases" not in data:
            new_map = _empty_label_map()
            for key in new_map:
                if key in data:
                    new_map[key] = data[key]
            _atomic_write(path, new_map)
            return new_map

        return data

    except Exception:
        lm = _empty_label_map()
        _atomic_write(path, lm)
        return lm


def save_label_map(label_map: dict, version_note: Optional[str] = None):
    label_map["version"] = datetime.utcnow().isoformat()
    if version_note:
        label_map["version_note"] = version_note
    _atomic_write(CONFIG["label_map_json"], label_map)


# ============================================================
# Public-facing sys_id utilities
# ============================================================

def register_sys_id(sys_id: str, alias: Optional[str], label_map: dict) -> None:
    """
    Register a canonical sys_id and (optional) alias.
    Idempotent, non-destructive.
    """
    can = normalize_sys_id(sys_id)
    if not can:
        return

    if can not in label_map["seen_sys_ids"]:
        label_map["seen_sys_ids"].append(can)
        label_map["seen_sys_ids"].sort()

    if alias:
        alias = alias.strip()
        if alias:
            label_map["aliases"].setdefault(can, [])
            if alias not in label_map["aliases"][can]:
                label_map["aliases"][can].append(alias)

    save_label_map(label_map, version_note="Registered sys_id/alias")


def resolve_to_sys_id(value: str, label_map: dict) -> str:
    """
    Resolve any input string or alias into a canonical sys_id.
    Return normalized input if not resolvable.
    """
    candidate = normalize_sys_id(value)
    if not candidate:
        return ""

    if candidate in set(label_map["seen_sys_ids"]):
        return candidate

    for can, alias_list in label_map["aliases"].items():
        for a in alias_list:
            if candidate == normalize_sys_id(a):
                return can

    return candidate


def is_unseen_sys_id(sys_id: str, label_map: dict) -> bool:
    resolved = resolve_to_sys_id(sys_id, label_map)
    return resolved not in set(label_map["seen_sys_ids"])


# ============================================================
# Folder name map (sys_id -> folder name)
# ============================================================

def load_folder_name_map() -> Dict[str, str]:
    """Load sys_id → human folder name mapping."""
    path = CONFIG.get("folder_name_map")
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {normalize_sys_id(k): v for k, v in data.items()}
    except Exception:
        return {}
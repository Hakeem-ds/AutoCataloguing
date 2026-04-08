import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from .config import CONFIG
from .file_utils import atomic_write


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
# Taxonomy-level normalisation (zero-padding + hierarchy)
# ============================================================

import re as _re


def _parse_segment(segment: str) -> tuple:
    """
    Split a SysID segment into (alpha_prefix, numeric_value).
    Examples:
        'LT000258' → ('LT', 258)
        '003'      → ('', 3)
        '01'       → ('', 1)
        'LT00032'  → ('LT', 32)
        'ABC'      → ('ABC', None)  # no numeric part
    """
    m = _re.match(r'^([A-Z]*)(\d+)$', segment, _re.IGNORECASE)
    if m:
        alpha, num_str = m.groups()
        return (alpha.upper(), int(num_str))
    return (segment.upper(), None)


def _canonicalize_segments(sysid: str) -> str:
    """
    Convert each segment to (alpha + int_value) for comparison.
    Strips leading zeros but preserves the numeric VALUE.
    'LT000258/003/004' → 'LT:258/3/4'
    'LT0258/3/4/14'    → 'LT:258/3/4/14'
    Both map to the same canonical form.
    """
    parts = sysid.split("/")
    result = []
    for p in parts:
        alpha, num = _parse_segment(p)
        if num is not None:
            result.append(f"{alpha}:{num}")
        else:
            result.append(alpha)
    return "/".join(result)


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance — used for root-segment typo detection."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


class TaxonomyIndex:
    """
    Pre-built index for fast taxonomy lookups.
    Supports exact match, numeric-canonical match, and edit-distance fallback.
    """

    def __init__(self, taxonomy_keys: set):
        self.taxonomy_keys = taxonomy_keys

        # Index 1: canonical (alpha:int) form → taxonomy key
        self.canonical_index = {}
        for tk in taxonomy_keys:
            canon = _canonicalize_segments(tk)
            if canon not in self.canonical_index:
                self.canonical_index[canon] = tk

        # Index 2: unique root segments for edit-distance search
        self.roots = {}  # root_segment → set of taxonomy keys starting with it
        for tk in taxonomy_keys:
            root = tk.split("/")[0]
            self.roots.setdefault(root, set()).add(tk)

    def resolve(self, raw_sysid: str) -> tuple:
        """
        Resolve a raw SysID to its canonical taxonomy form.

        Resolution order prioritises SPECIFICITY — keep the full path
        whenever possible, only collapse (walk-up) as a last resort.

        Chain:
          1. Exact match
          2. Zero-pad full match (LT0258/003/004/014 → LT000258/003/004/014)
          3. Edit-distance full match (LT00032/003/001 → LT000320/003/001)
          4. Zero-pad + walk-up (LT000266/001/025/002/001 → .../002)
          5. Edit-distance + walk-up
          6. Raw walk-up (last resort — loses specificity)
          7. Passthrough

        Returns (resolved_sysid, method) where method is one of:
            'exact', 'zeropad', 'editdist', 'zeropad_walkup',
            'editdist_walkup', 'walkup', 'passthrough'
        """
        normed = normalize_sys_id(raw_sysid)
        if not normed:
            return normed, "empty"

        # 1. Exact match
        if normed in self.taxonomy_keys:
            return normed, "exact"

        parts = normed.split("/")

        # 2. Zero-pad full match (numeric-canonical comparison)
        canon = _canonicalize_segments(normed)
        if canon in self.canonical_index:
            return self.canonical_index[canon], "zeropad"

        # 3. Edit-distance full match on root (handles LT00032 → LT000320)
        input_root = parts[0]
        best_tk = None
        best_dist = 3  # max allowed

        for tax_root, tax_keys in self.roots.items():
            dist = _edit_distance(input_root, tax_root)
            if dist >= best_dist:
                continue
            for tk in tax_keys:
                tk_parts = tk.split("/")
                if len(parts) > len(tk_parts):
                    continue
                # Compare child segments using canonical form
                segments_match = all(
                    _canonicalize_segments(parts[i]) == _canonicalize_segments(tk_parts[i])
                    for i in range(1, min(len(parts), len(tk_parts)))
                )
                if segments_match:
                    resolved = "/".join(tk_parts[:len(parts)])
                    if resolved in self.taxonomy_keys:
                        best_tk = resolved
                        best_dist = dist
                        break

        if best_tk:
            return best_tk, "editdist"

        # 4. Zero-pad + walk-up (file-level after padding)
        canon_parts = canon.split("/")
        for depth in range(len(canon_parts) - 1, 0, -1):
            candidate = "/".join(canon_parts[:depth])
            if candidate in self.canonical_index:
                return self.canonical_index[candidate], "zeropad_walkup"

        # 5. Edit-distance + walk-up
        for tax_root, tax_keys in self.roots.items():
            dist = _edit_distance(input_root, tax_root)
            if dist >= 3:
                continue
            for tk in tax_keys:
                tk_parts = tk.split("/")
                for depth in range(min(len(parts), len(tk_parts)) - 1, 0, -1):
                    segments_match = all(
                        _canonicalize_segments(parts[i]) == _canonicalize_segments(tk_parts[i])
                        for i in range(1, depth)
                    )
                    if segments_match:
                        candidate = "/".join(tk_parts[:depth])
                        if candidate in self.taxonomy_keys:
                            return candidate, "editdist_walkup"

        # 6. Raw walk-up (last resort — loses specificity)
        for depth in range(len(parts) - 1, 0, -1):
            candidate = "/".join(parts[:depth])
            if candidate in self.taxonomy_keys:
                return candidate, "walkup"

        # 6. Parent walk-up — LAST resort (file-level index removal)
        #    Only fires when padding fixes didn't help. This is lossy
        #    (drops the last segment), so it runs after all corrections.
        for depth in range(len(parts) - 1, 0, -1):
            candidate = "/".join(parts[:depth])
            if candidate in self.taxonomy_keys:
                return candidate, "walkup"

        # 7. Passthrough — genuinely novel label
        return normed, "passthrough"


# Module-level singleton — lazily built on first call
_taxonomy_index_cache: dict = {}


def _get_taxonomy_index(taxonomy_keys: set) -> TaxonomyIndex:
    """Get or build the TaxonomyIndex for a given taxonomy key set."""
    cache_key = id(taxonomy_keys)
    if cache_key not in _taxonomy_index_cache:
        _taxonomy_index_cache[cache_key] = TaxonomyIndex(taxonomy_keys)
    return _taxonomy_index_cache[cache_key]


def normalise_to_taxonomy(
    raw_sysid: str,
    taxonomy_keys: set,
) -> str:
    """
    Normalise a SysID to the canonical taxonomy form.

    Resolution chain (stops at first match):
      1. Exact match in taxonomy.
      2. Parent walk-up (removes file-level index: .../007 → .../002).
      3. Zero-pad match (LT0258 → LT000258, /01/ → /001/).
      4. Zero-pad + walk-up combined.
      5. Edit-distance on root segment (LT00032 → LT000320, max dist 2)
         with remaining segment validation.
      6. Passthrough — return normalised input unchanged.

    Returns the canonical taxonomy SysID, or the normalised input if no match.
    """
    idx = _get_taxonomy_index(taxonomy_keys)
    resolved, _ = idx.resolve(raw_sysid)
    return resolved


def normalise_to_taxonomy_verbose(
    raw_sysid: str,
    taxonomy_keys: set,
) -> tuple:
    """
    Like normalise_to_taxonomy but also returns the resolution method.
    Useful for diagnostics and the taxonomy manager typo detection UI.

    Returns (resolved_sysid, method).
    """
    idx = _get_taxonomy_index(taxonomy_keys)
    return idx.resolve(raw_sysid)


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


def load_label_map() -> dict:
    """Load the sys_id/alias map. Auto-create if missing."""
    path = CONFIG["label_map_json"]

    if not os.path.exists(path):
        lm = _empty_label_map()
        atomic_write(path, lm)
        return lm

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "seen_sys_ids" not in data or "aliases" not in data:
            new_map = _empty_label_map()
            for key in new_map:
                if key in data:
                    new_map[key] = data[key]
            atomic_write(path, new_map)
            return new_map

        return data

    except Exception:
        lm = _empty_label_map()
        atomic_write(path, lm)
        return lm


def save_label_map(label_map: dict, version_note: Optional[str] = None):
    label_map["version"] = datetime.utcnow().isoformat()
    if version_note:
        label_map["version_note"] = version_note
    atomic_write(CONFIG["label_map_json"], label_map)


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

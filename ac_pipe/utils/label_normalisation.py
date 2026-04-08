# utils/label_normalisation.py — Standalone label normalisation for the training pipeline.
#
# Extracted from streamlit_app/src/core/label_map.py so the pipeline has
# ZERO dependency on the Streamlit app source tree.  This makes it work
# reliably on classic clusters, serverless, and containers.
#
# Functions:
#   normalize_sys_id()        — format-level cleanup
#   resolve_hierarchical_sys_id() — walk-up matching
#   normalise_to_taxonomy()   — full resolution chain (exact → zeropad → editdist → walkup)

from __future__ import annotations

import re as _re
from typing import Any, Optional


# ================================================================
# Format-level normalisation
# ================================================================

def normalize_sys_id(value: Any) -> str:
    """
    Format-level normalisation:
       - convert to string, trim, uppercase
       - replace spaces/underscores → hyphens
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


# ================================================================
# Hierarchical walk-up
# ================================================================

def resolve_hierarchical_sys_id(raw_sys_id: str, valid_folders: set[str]) -> Optional[str]:
    """
    Given a SYSID like "A/B/C/D/E/F", progressively try
    A/B/C/D/E/F, A/B/C/D/E, ..., A/B, A.
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


# ================================================================
# Internal helpers for zero-padding and edit-distance
# ================================================================

def _parse_segment(segment: str) -> tuple:
    m = _re.match(r'^([A-Z]*)(\d+)$', segment, _re.IGNORECASE)
    if m:
        alpha, num_str = m.groups()
        return (alpha.upper(), int(num_str))
    return (segment.upper(), None)


def _canonicalize_segments(sysid: str) -> str:
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


# ================================================================
# TaxonomyIndex — pre-built index for fast lookups
# ================================================================

class TaxonomyIndex:
    def __init__(self, taxonomy_keys: set):
        self.taxonomy_keys = taxonomy_keys
        self.canonical_index = {}
        for tk in taxonomy_keys:
            canon = _canonicalize_segments(tk)
            if canon not in self.canonical_index:
                self.canonical_index[canon] = tk
        self.roots = {}
        for tk in taxonomy_keys:
            root = tk.split("/")[0]
            self.roots.setdefault(root, set()).add(tk)

    def resolve(self, raw_sysid: str) -> tuple:
        normed = normalize_sys_id(raw_sysid)
        if not normed:
            return normed, "empty"
        if normed in self.taxonomy_keys:
            return normed, "exact"
        parts = normed.split("/")
        canon = _canonicalize_segments(normed)
        if canon in self.canonical_index:
            return self.canonical_index[canon], "zeropad"
        input_root = parts[0]
        best_tk, best_dist = None, 3
        for tax_root, tax_keys in self.roots.items():
            dist = _edit_distance(input_root, tax_root)
            if dist >= best_dist:
                continue
            for tk in tax_keys:
                tk_parts = tk.split("/")
                if len(parts) > len(tk_parts):
                    continue
                segments_match = all(
                    _canonicalize_segments(parts[i]) == _canonicalize_segments(tk_parts[i])
                    for i in range(1, min(len(parts), len(tk_parts)))
                )
                if segments_match:
                    resolved = "/".join(tk_parts[:len(parts)])
                    if resolved in self.taxonomy_keys:
                        best_tk, best_dist = resolved, dist
                        break
        if best_tk:
            return best_tk, "editdist"
        canon_parts = canon.split("/")
        for depth in range(len(canon_parts) - 1, 0, -1):
            candidate = "/".join(canon_parts[:depth])
            if candidate in self.canonical_index:
                return self.canonical_index[candidate], "zeropad_walkup"
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
        for depth in range(len(parts) - 1, 0, -1):
            candidate = "/".join(parts[:depth])
            if candidate in self.taxonomy_keys:
                return candidate, "walkup"
        return normed, "passthrough"


# Module-level singleton
_taxonomy_index_cache: dict = {}


def _get_taxonomy_index(taxonomy_keys: set) -> TaxonomyIndex:
    cache_key = id(taxonomy_keys)
    if cache_key not in _taxonomy_index_cache:
        _taxonomy_index_cache[cache_key] = TaxonomyIndex(taxonomy_keys)
    return _taxonomy_index_cache[cache_key]


def normalise_to_taxonomy(raw_sysid: str, taxonomy_keys: set) -> str:
    """Normalise a SysID to the canonical taxonomy form. Zero external deps."""
    idx = _get_taxonomy_index(taxonomy_keys)
    resolved, _ = idx.resolve(raw_sysid)
    return resolved

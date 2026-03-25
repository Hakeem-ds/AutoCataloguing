import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime

from core.file_utils import load_csv_or_excel
from core.config import CONFIG
from core.label_map import (
    normalize_sys_id,
    load_folder_name_map,
    load_label_map,
    register_sys_id,
)

st.set_page_config(page_title="Taxonomy Manager", page_icon="🗂️", layout="wide")
st.title("🗂️ Taxonomy Manager")

with st.expander("📘 What this page does", expanded=True):
    st.markdown("""
Manage your **taxonomy** (SysID → Folder Name) safely and easily.

**You can:**
- Browse & search entries  
- Add/update SysIDs and names  
- **Bulk import** a CSV with preview and validation  
- **Auto-create missing parents** in a hierarchy (e.g., create `A/B` before `A/B/C`)  
- Manage **Aliases** (common alternative labels)  
- See an **Audit Log** of every change  
- Export the taxonomy (JSON/CSV)
""")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _atomic_write(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def _parents(sysid: str) -> list[str]:
    """Return all parent prefixes of a hierarchical SysID, excluding itself."""
    parts = sysid.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts))]

def _placeholder_name(sysid: str) -> str:
    """Generate a human placeholder name for auto-created parent nodes."""
    last = sysid.split("/")[-1] if sysid else "ROOT"
    return f"{last} (auto)"

def _write_audit(action: str, sys_id: str, folder_name: str, details: dict | None = None):
    """Append an audit row to artifacts/taxonomy_audit_log.csv (create if missing)."""
    audit_path = os.path.join(CONFIG["artifacts_dir"], "taxonomy_audit_log.csv")
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "sys_id": sys_id,
        "folder_name": folder_name,
        "details": json.dumps(details or {}, ensure_ascii=False),
    }
    if os.path.exists(audit_path):
        try:
            df = pd.read_csv(audit_path)
        except Exception:
            df = pd.DataFrame(columns=row.keys())
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(audit_path, index=False)
    else:
        pd.DataFrame([row]).to_csv(audit_path, index=False)

def _save_csv_taxonomy(folder_name_map: dict):
    """Upsert the CSV taxonomy (CONFIG['folder_mapping_csv']) from the current map."""
    csv_path = CONFIG.get("folder_mapping_csv")
    if not csv_path:
        return
    rows = [{"sys_id": k, "folder_name": v, "updated_at": datetime.utcnow().isoformat()} for k, v in folder_name_map.items()]
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["sys_id"], keep="last")
    df.to_csv(csv_path, index=False)


# -------------------------------------------------------------------
# Load current taxonomy
# -------------------------------------------------------------------
folder_name_map = load_folder_name_map()  # {sys_id -> folder_name} with normalized keys
valid_sysids = set(folder_name_map.keys())

# -------------------------------------------------------------------
# Toolbar: Export / Refresh
# -------------------------------------------------------------------
tool_c1, tool_c2, tool_c3 = st.columns(3)
with tool_c1:
    json_bytes = json.dumps(folder_name_map, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button("⬇️ Download JSON", data=json_bytes, file_name="folder_name_map.json", mime="application/json")
with tool_c2:
    csv_df = pd.DataFrame([{"sys_id": k, "folder_name": v} for k, v in sorted(folder_name_map.items())])
    st.download_button(
        "⬇️ Download CSV",
        data=csv_df.to_csv(index=False).encode("utf-8"),
        file_name="folder_mapping.csv",
        mime="text/csv"
    )
with tool_c3:
    if st.button("🔄 Refresh"):
        _safe_rerun()

# -------------------------------------------------------------------
# Search / Browse
# -------------------------------------------------------------------
st.subheader("🔎 Search / Browse")
query = st.text_input("Search SysID or Folder Name", "")
view = folder_name_map
if query.strip():
    q = query.strip().lower()
    view = {k: v for k, v in folder_name_map.items() if q in k.lower() or q in (v or "").lower()}

df_view = pd.DataFrame([{"SysID": k, "Folder Name": v} for k, v in sorted(view.items())])
st.dataframe(df_view, use_container_width=True, height=320)

# -------------------------------------------------------------------
# Add / Update single entry (with parent enforcement)
# -------------------------------------------------------------------
st.write("---")
st.subheader("➕ Add or Update a Taxonomy Entry")

with st.form("add_update_form"):
    new_sysid_raw = st.text_input("SysID (e.g., A/B/C/001)", "")
    new_name = st.text_input("Folder Name", "")
    ensure_parents = st.checkbox("Auto-create missing parents (recommended)", value=True)
    also_register = st.checkbox("Also register this SysID in the label map (recommended)", value=True)
    submitted = st.form_submit_button("Save")

if submitted:
    can = normalize_sys_id(new_sysid_raw)
    if not can:
        st.error("Please enter a valid SysID.")
    elif not new_name.strip():
        st.error("Please enter a folder name.")
    else:
        # Parent enforcement
        missing_parents = [p for p in _parents(can) if p and p not in folder_name_map]
        created_parents = []
        if missing_parents and ensure_parents:
            # Create in increasing depth order
            for p in sorted(missing_parents, key=lambda x: len(x.split("/"))):
                folder_name_map[p] = _placeholder_name(p)
                created_parents.append(p)
                _write_audit("create_parent", p, folder_name_map[p], {"reason": "auto-create-parents"})
                if also_register:
                    lm = load_label_map()
                    register_sys_id(sys_id=p, alias=None, label_map=lm)

        # Upsert main entry
        folder_name_map[can] = new_name.strip()
        _atomic_write(CONFIG["folder_name_map"], folder_name_map)
        _save_csv_taxonomy(folder_name_map)
        _write_audit("upsert_entry", can, new_name.strip(), {"created_parents": created_parents})

        if also_register:
            lm = load_label_map()
            register_sys_id(sys_id=can, alias=None, label_map=lm)

        if missing_parents and not ensure_parents:
            st.warning(f"Missing parents not created: {missing_parents}. Consider enabling 'Auto-create missing parents'.")

        st.success(f"Saved SysID **{can}** → **{new_name.strip()}**")
        _safe_rerun()

# -------------------------------------------------------------------
# Quick Update Folder Name
# -------------------------------------------------------------------
st.write("---")
st.subheader("✏️ Quick Update Folder Name")

with st.form("quick_update_form"):
    if folder_name_map:
        existing_id = st.selectbox("Choose SysID", options=sorted(list(folder_name_map.keys())))
        new_name2 = st.text_input("New Folder Name", value=folder_name_map.get(existing_id, ""))
        upd = st.form_submit_button("Update")
    else:
        existing_id, new_name2, upd = None, None, False
        st.info("No taxonomy entries yet. Add one above.")

if upd and existing_id:
    folder_name_map[existing_id] = new_name2.strip()
    _atomic_write(CONFIG["folder_name_map"], folder_name_map)
    _save_csv_taxonomy(folder_name_map)
    _write_audit("rename_entry", existing_id, new_name2.strip(), {})
    st.success(f"Updated **{existing_id}** → **{new_name2.strip()}**")
    _safe_rerun()

# -------------------------------------------------------------------
# Bulk Import with Preview & Validation
# -------------------------------------------------------------------
st.write("---")
st.subheader("📥 Bulk Import (CSV)")

st.markdown("Upload a CSV with columns **sys_id** and **folder_name**. We'll normalize IDs, show a preview, validate parents, and apply changes safely.")
bulk_file = st.file_uploader("Upload CSV", type=["csv", "xlsx", "xls"], key="bulk_taxonomy_upload")

if bulk_file is not None:
    try:
        incoming = load_csv_or_excel(bulk_file)
        if not isinstance(incoming, pd.DataFrame):
            raise ValueError("Uploaded file did not produce a DataFrame.")
    except Exception as e:
        st.error(f"Could not read CSV or Excel file: {e}")
        incoming = pd.DataFrame(columns=["sys_id", "folder_name"])

    # Validate columns
    required_cols = {"sys_id", "folder_name"}
    if not required_cols.issubset(set(c.lower() for c in incoming.columns)):
        st.error("CSV must include columns: sys_id, folder_name")
    else:
        # Normalize & coalesce duplicates (keep last occurrence)
        sys_col = [c for c in incoming.columns if c.lower() == "sys_id"][0]
        name_col = [c for c in incoming.columns if c.lower() == "folder_name"][0]

        incoming["sys_norm"] = incoming[sys_col].astype(str).apply(normalize_sys_id)
        incoming["name_norm"] = incoming[name_col].astype(str).apply(lambda s: s.strip())

        # Drop empty
        incoming = incoming[(incoming["sys_norm"] != "") & (incoming["name_norm"] != "")]
        # Keep last occurrence per sys_norm
        incoming = incoming.drop_duplicates(subset=["sys_norm"], keep="last")

        # Build preview with missing parents info
        preview_rows = []
        for _, r in incoming.iterrows():
            sid = r["sys_norm"]
            name = r["name_norm"]
            parents_missing = [p for p in _parents(sid) if p and p not in folder_name_map]
            preview_rows.append({
                "SysID": sid,
                "Folder Name": name,
                "Exists": sid in folder_name_map,
                "Missing Parents Count": len(parents_missing),
                "Missing Parents": "; ".join(parents_missing),
            })

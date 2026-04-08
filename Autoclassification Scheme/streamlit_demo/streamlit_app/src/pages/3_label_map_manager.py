import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime

from core.file_utils import load_csv_or_excel, atomic_write, safe_rerun
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
folder_name_map = load_folder_name_map()
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
        safe_rerun()

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
        missing_parents = [p for p in _parents(can) if p and p not in folder_name_map]
        created_parents = []
        if missing_parents and ensure_parents:
            for p in sorted(missing_parents, key=lambda x: len(x.split("/"))):
                folder_name_map[p] = _placeholder_name(p)
                created_parents.append(p)
                _write_audit("create_parent", p, folder_name_map[p], {"reason": "auto-create-parents"})
                if also_register:
                    lm = load_label_map()
                    register_sys_id(sys_id=p, alias=None, label_map=lm)

        folder_name_map[can] = new_name.strip()
        atomic_write(CONFIG["folder_name_map"], folder_name_map)
        _save_csv_taxonomy(folder_name_map)
        _write_audit("upsert_entry", can, new_name.strip(), {"created_parents": created_parents})

        if also_register:
            lm = load_label_map()
            register_sys_id(sys_id=can, alias=None, label_map=lm)

        if missing_parents and not ensure_parents:
            st.warning(f"Missing parents not created: {missing_parents}. Consider enabling 'Auto-create missing parents'.")

        st.success(f"Saved SysID **{can}** → **{new_name.strip()}**")
        safe_rerun()

# -------------------------------------------------------------------
# Quick Update Folder Name
# -------------------------------------------------------------------
st.write("---")
st.subheader("✏️ Quick Update Folder Name")

with st.form("quick_update_form"):
    if folder_name_map:
        existing_id = st.selectbox("Choose SysID", options=sorted(list(folder_name_map.keys())))
        new_name2 = st.text_input("New Folder Name", value=folder_name_map.get(existing_id, ""))
    else:
        existing_id, new_name2 = None, None
        st.info("No taxonomy entries yet. Add one above.")
    upd = st.form_submit_button("Update")

if upd and existing_id:
    folder_name_map[existing_id] = new_name2.strip()
    atomic_write(CONFIG["folder_name_map"], folder_name_map)
    _save_csv_taxonomy(folder_name_map)
    _write_audit("rename_entry", existing_id, new_name2.strip(), {})
    st.success(f"Updated **{existing_id}** → **{new_name2.strip()}**")
    safe_rerun()

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

    required_cols = {"sys_id", "folder_name"}
    if not required_cols.issubset(set(c.lower() for c in incoming.columns)):
        st.error("CSV must include columns: sys_id, folder_name")
    else:
        sys_col = [c for c in incoming.columns if c.lower() == "sys_id"][0]
        name_col = [c for c in incoming.columns if c.lower() == "folder_name"][0]

        incoming["sys_norm"] = incoming[sys_col].astype(str).apply(normalize_sys_id)
        incoming["name_norm"] = incoming[name_col].astype(str).apply(lambda s: s.strip())

        incoming = incoming[(incoming["sys_norm"] != "") & (incoming["name_norm"] != "")]
        incoming = incoming.drop_duplicates(subset=["sys_norm"], keep="last")

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

        preview_df = pd.DataFrame(preview_rows)
        st.write("### Preview")
        st.dataframe(preview_df, use_container_width=True, height=280)

        auto_parents = st.checkbox("Auto-create missing parents during import", value=True, key="bulk_auto_parents")
        also_reg_bulk = st.checkbox("Register imported SysIDs in label map", value=True, key="bulk_also_register")

        if st.button("✅ Apply Bulk Import", key="bulk_apply_btn"):
            count_new = 0
            count_updated = 0
            count_parents = 0

            for _, r in incoming.iterrows():
                sid = r["sys_norm"]
                name = r["name_norm"]

                if auto_parents:
                    for p in sorted(_parents(sid), key=lambda x: len(x.split("/"))):
                        if p and p not in folder_name_map:
                            folder_name_map[p] = _placeholder_name(p)
                            count_parents += 1
                            _write_audit("create_parent", p, folder_name_map[p], {"reason": "bulk-import-parents"})
                            if also_reg_bulk:
                                lm = load_label_map()
                                register_sys_id(sys_id=p, alias=None, label_map=lm)

                is_new = sid not in folder_name_map
                folder_name_map[sid] = name
                if is_new:
                    count_new += 1
                else:
                    count_updated += 1
                _write_audit("bulk_upsert", sid, name, {})

                if also_reg_bulk:
                    lm = load_label_map()
                    register_sys_id(sys_id=sid, alias=None, label_map=lm)

            atomic_write(CONFIG["folder_name_map"], folder_name_map)
            _save_csv_taxonomy(folder_name_map)

            st.success(
                f"Bulk import complete: **{count_new}** new, **{count_updated}** updated, "
                f"**{count_parents}** parents auto-created."
            )
            safe_rerun()

# -------------------------------------------------------------------
# Alias Manager
# -------------------------------------------------------------------
st.write("---")
st.subheader("🏷️ Alias Manager")

lm = load_label_map()
aliases = lm.get("aliases", {})

if aliases:
    alias_rows = []
    for can, als in aliases.items():
        for a in als:
            alias_rows.append({"Canonical SysID": can, "Alias": a})
    st.dataframe(pd.DataFrame(alias_rows), use_container_width=True, height=200)
else:
    st.info("No aliases defined yet.")

with st.form("alias_form"):
    alias_sysid = st.text_input("SysID to attach alias to", "")
    alias_val = st.text_input("Alias value", "")
    alias_sub = st.form_submit_button("Add Alias")

if alias_sub:
    can = normalize_sys_id(alias_sysid)
    if not can:
        st.error("Enter a valid SysID.")
    elif not alias_val.strip():
        st.error("Enter a valid alias.")
    else:
        lm = load_label_map()
        register_sys_id(sys_id=can, alias=alias_val.strip(), label_map=lm)
        st.success(f"Alias '{alias_val.strip()}' added for **{can}**.")
        safe_rerun()

# -------------------------------------------------------------------
# Audit Log
# -------------------------------------------------------------------
st.write("---")
st.subheader("📜 Audit Log")

audit_path = os.path.join(CONFIG["artifacts_dir"], "taxonomy_audit_log.csv")
if os.path.exists(audit_path):
    try:
        audit_df = pd.read_csv(audit_path)
        st.dataframe(audit_df.tail(50), use_container_width=True, height=300)
    except Exception:
        st.info("Could not read audit log.")
else:
    st.info("No audit log yet. Changes will be logged here.")


# ===================================================================
# SysID Resolver & Typo Detection
# ===================================================================
st.write("---")
st.header("🔍 SysID Resolver & Typo Detection")

from core.label_map import normalise_to_taxonomy_verbose, TaxonomyIndex, _canonicalize_segments

_METHOD_COLOURS = {
    "exact":          "🟢",
    "zeropad":        "🟡",
    "zeropad_walkup": "🟠",
    "editdist":       "🔴",
    "editdist_walkup":"🔴",
    "walkup":         "🟠",
    "passthrough":    "⚪",
    "empty":          "⚫",
}

_METHOD_DESCRIPTIONS = {
    "exact":          "Exact match — already in taxonomy",
    "zeropad":        "Zero-padding fix (e.g. /01/ → /001/, LT0258 → LT000258)",
    "zeropad_walkup": "Zero-padding + parent walk-up (file-level index collapsed)",
    "editdist":       "Edit-distance root match (e.g. LT00032 → LT000320)",
    "editdist_walkup":"Edit-distance + parent walk-up",
    "walkup":         "Parent walk-up only (file-level index removed)",
    "passthrough":    "No match found — novel label or unrecognised format",
    "empty":          "Empty input",
}

resolver_tab, batch_tab, audit_tab, stats_tab, kb_tab = st.tabs([
    "🔎 Live Resolver",
    "📋 Batch Resolver",
    "🩺 Data Audit",
    "📊 Resolution Stats",
    "📚 Knowledge Base Search",
])

# ── Tab 1: Live Resolver ──────────────────────────────────────────
with resolver_tab:
    st.markdown("""
    Type or paste a SysID to see how the normalisation engine resolves it.
    Shows the **resolution method**, the **canonical taxonomy SysID**, and the **folder name**.
    """)

    live_input = st.text_input("Enter SysID to resolve", "", key="live_resolve_input")

    if live_input.strip():
        resolved, method = normalise_to_taxonomy_verbose(live_input.strip(), valid_sysids)
        name = folder_name_map.get(resolved, "—")
        icon = _METHOD_COLOURS.get(method, "❓")
        desc = _METHOD_DESCRIPTIONS.get(method, method)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Resolved SysID", resolved)
        with col2:
            st.metric("Folder Name", name)

        st.info(f"{icon} **Method: {method}** — {desc}")

        if method != "exact" and method != "passthrough" and method != "empty":
            st.success(f"✅ Correction: `{live_input.strip()}` → `{resolved}`")
            # Show the canonical forms for transparency
            with st.expander("🔬 Resolution Details"):
                st.write(f"**Input (normalised):** `{normalize_sys_id(live_input.strip())}`")
                st.write(f"**Input (canonical):** `{_canonicalize_segments(normalize_sys_id(live_input.strip()))}`")
                if resolved in folder_name_map:
                    st.write(f"**Taxonomy (canonical):** `{_canonicalize_segments(resolved)}`")
                # Show siblings at same level
                if "/" in resolved:
                    parent = "/".join(resolved.split("/")[:-1])
                    siblings = [k for k in folder_name_map if k.startswith(parent + "/") and k.count("/") == resolved.count("/")]
                    if siblings:
                        st.write(f"**Siblings under `{parent}`:**")
                        sib_data = [{"SysID": s, "Name": folder_name_map.get(s, "—")} for s in sorted(siblings)]
                        st.dataframe(pd.DataFrame(sib_data), use_container_width=True, height=200)

        elif method == "passthrough":
            st.warning("⚠️ No taxonomy match found. This is either a genuinely new label or a format the resolver can't fix automatically.")
            # Suggest closest matches by canonical form
            normed = normalize_sys_id(live_input.strip())
            canon = _canonicalize_segments(normed)
            idx = TaxonomyIndex(valid_sysids)
            # Show entries sharing the same root
            root = normed.split("/")[0]
            similar = [k for k in folder_name_map if k.split("/")[0] == root]
            if similar:
                st.write(f"**Entries sharing root `{root}`:**")
                sim_data = [{"SysID": k, "Name": folder_name_map.get(k, "—")} for k in sorted(similar)[:20]]
                st.dataframe(pd.DataFrame(sim_data), use_container_width=True, height=200)

# ── Tab 2: Batch Resolver ─────────────────────────────────────────
with batch_tab:
    st.markdown("""
    Paste multiple SysIDs (one per line) or upload a CSV file to resolve them in bulk.
    Download the results as a CSV with resolution details.
    """)

    input_mode = st.radio("Input mode", ["Paste SysIDs", "Upload CSV"], horizontal=True, key="batch_input_mode")

    batch_sysids = []

    if input_mode == "Paste SysIDs":
        raw_text = st.text_area(
            "Paste SysIDs (one per line)",
            height=200,
            placeholder="LT000276/01/014\nLT0258/003/004/014\nLT00032/003/001",
            key="batch_paste_area",
        )
        if raw_text.strip():
            batch_sysids = [s.strip() for s in raw_text.strip().splitlines() if s.strip()]

    else:
        batch_file = st.file_uploader("Upload CSV (column: sys_id or SysID)", type=["csv"], key="batch_resolve_upload")
        if batch_file is not None:
            try:
                batch_df_in = pd.read_csv(batch_file)
                # Find the SysID column
                sid_col = None
                for c in batch_df_in.columns:
                    if c.lower().replace(" ", "").replace("_", "") in ("sysid", "newsysid", "systemid", "reference"):
                        sid_col = c
                        break
                if sid_col is None:
                    sid_col = batch_df_in.columns[0]
                    st.info(f"Using first column `{sid_col}` as SysID source")
                batch_sysids = batch_df_in[sid_col].astype(str).tolist()
                batch_sysids = [s.strip() for s in batch_sysids if s.strip() and s.strip().lower() != "nan"]
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    if batch_sysids:
        st.write(f"**{len(batch_sysids)} SysIDs to resolve**")

        results = []
        for raw in batch_sysids:
            resolved, method = normalise_to_taxonomy_verbose(raw, valid_sysids)
            name = folder_name_map.get(resolved, "—")
            icon = _METHOD_COLOURS.get(method, "❓")
            results.append({
                "Input": raw,
                "Resolved": resolved,
                "Folder Name": name,
                "Method": f"{icon} {method}",
                "Changed": "Yes" if normalize_sys_id(raw) != resolved else "No",
            })

        results_df = pd.DataFrame(results)

        # Summary stats
        method_counts = results_df["Method"].value_counts()
        changed_count = (results_df["Changed"] == "Yes").sum()
        passthrough_count = results_df["Method"].str.contains("passthrough").sum()

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total", len(results_df))
        mc2.metric("Corrected", changed_count)
        mc3.metric("Unresolved", passthrough_count)

        st.dataframe(results_df, use_container_width=True, height=400)

        # Download
        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Resolution Report",
            data=csv_out,
            file_name="sysid_resolution_report.csv",
            mime="text/csv",
        )

# ── Tab 3: Data Audit ────────────────────────────────────────────
with audit_tab:
    st.markdown("""
    Scans **training data** and **feedback corrections** for SysIDs that don't exactly
    match the taxonomy. Shows what would change under normalisation, with class-count
    impact analysis.
    """)

    if st.button("🩺 Run Data Audit", key="run_data_audit"):
        audit_results = []

        # Scan training_data.csv
        td_path = CONFIG.get("training_data_csv")
        if td_path and os.path.exists(td_path):
            td_df = pd.read_csv(td_path, dtype=str, keep_default_na=False)
            sid_col = "SysID" if "SysID" in td_df.columns else td_df.columns[0]
            for raw in td_df[sid_col].unique():
                resolved, method = normalise_to_taxonomy_verbose(str(raw).strip(), valid_sysids)
                if method != "exact" and method != "empty":
                    count = len(td_df[td_df[sid_col] == raw])
                    audit_results.append({
                        "Source": "training_data.csv",
                        "Raw SysID": raw,
                        "Resolved": resolved,
                        "Folder Name": folder_name_map.get(resolved, "—"),
                        "Method": f"{_METHOD_COLOURS.get(method, '❓')} {method}",
                        "Rows Affected": count,
                    })

        # Scan feedback.csv
        fb_path = CONFIG.get("feedback_csv")
        if fb_path and os.path.exists(fb_path):
            fb_df = pd.read_csv(fb_path, dtype=str, keep_default_na=False)
            for col in ["Correct SysID", "Predicted SysID"]:
                if col in fb_df.columns:
                    for raw in fb_df[col].unique():
                        resolved, method = normalise_to_taxonomy_verbose(str(raw).strip(), valid_sysids)
                        if method != "exact" and method != "empty":
                            count = len(fb_df[fb_df[col] == raw])
                            audit_results.append({
                                "Source": f"feedback.csv ({col})",
                                "Raw SysID": raw,
                                "Resolved": resolved,
                                "Folder Name": folder_name_map.get(resolved, "—"),
                                "Method": f"{_METHOD_COLOURS.get(method, '❓')} {method}",
                                "Rows Affected": count,
                            })

        # Scan used_training_data from MLflow (promoted model)
        try:
            import mlflow
            from core.model_registry import load_registry
            reg = load_registry()
            promoted = reg.get("__meta__", {}).get("promoted")
            if promoted and promoted in reg:
                uri = reg[promoted].get("mlflow_model_uri", "")
                if uri:
                    run_id = uri.split("/")[1]
                    client = mlflow.tracking.MlflowClient()
                    run = client.get_run(run_id)
                    parent_id = run.data.tags.get("mlflow.parentRunId", run_id)
                    from mlflow.artifacts import download_artifacts
                    csv_art = download_artifacts(f"runs:/{parent_id}/artifacts/used_training_data.csv")
                    utd = pd.read_csv(csv_art, dtype=str, keep_default_na=False)
                    sid_col2 = "SysID" if "SysID" in utd.columns else utd.columns[0]
                    before_classes = utd[sid_col2].nunique()
                    utd["_resolved"] = utd[sid_col2].apply(
                        lambda s: normalise_to_taxonomy_verbose(str(s).strip(), valid_sysids)[0]
                    )
                    after_classes = utd["_resolved"].nunique()
                    for raw in utd[sid_col2].unique():
                        resolved, method = normalise_to_taxonomy_verbose(str(raw).strip(), valid_sysids)
                        if method != "exact" and method != "empty":
                            count = len(utd[utd[sid_col2] == raw])
                            audit_results.append({
                                "Source": "MLflow (used_training_data)",
                                "Raw SysID": raw,
                                "Resolved": resolved,
                                "Folder Name": folder_name_map.get(resolved, "—"),
                                "Method": f"{_METHOD_COLOURS.get(method, '❓')} {method}",
                                "Rows Affected": count,
                            })
                    st.info(f"📊 **MLflow training data impact:** {before_classes} classes → {after_classes} classes ({before_classes - after_classes} eliminated)")
        except Exception as e:
            st.caption(f"Could not scan MLflow training data: {e}")

        if audit_results:
            audit_df = pd.DataFrame(audit_results)
            total_affected = audit_df["Rows Affected"].sum()
            unique_corrections = len(audit_df)

            ac1, ac2, ac3 = st.columns(3)
            ac1.metric("Non-Exact Labels", unique_corrections)
            ac2.metric("Total Rows Affected", int(total_affected))
            ac3.metric("Unique Methods", audit_df["Method"].nunique())

            st.dataframe(audit_df, use_container_width=True, height=400)

            csv_audit = audit_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Audit Report",
                data=csv_audit,
                file_name="taxonomy_audit_report.csv",
                mime="text/csv",
            )
        else:
            st.success("✅ All SysIDs in training data and feedback match the taxonomy exactly.")

# ── Tab 4: Resolution Statistics ──────────────────────────────────
with stats_tab:
    st.markdown("""
    Aggregate statistics on how the normalisation engine resolves labels
    across all data sources. Helps identify **systemic patterns** (e.g.
    many zero-pad issues in a particular collection).
    """)

    if st.button("📊 Compute Resolution Stats", key="compute_res_stats"):
        all_sysids = []

        # Gather from training_data.csv
        td_path = CONFIG.get("training_data_csv")
        if td_path and os.path.exists(td_path):
            td_df = pd.read_csv(td_path, dtype=str, keep_default_na=False)
            sid_col = "SysID" if "SysID" in td_df.columns else td_df.columns[0]
            all_sysids.extend(td_df[sid_col].astype(str).tolist())

        # Gather from feedback.csv
        fb_path = CONFIG.get("feedback_csv")
        if fb_path and os.path.exists(fb_path):
            fb_df = pd.read_csv(fb_path, dtype=str, keep_default_na=False)
            for col in ["Correct SysID", "Predicted SysID"]:
                if col in fb_df.columns:
                    all_sysids.extend(fb_df[col].astype(str).tolist())

        if all_sysids:
            method_list = []
            collection_issues = {}
            for raw in all_sysids:
                resolved, method = normalise_to_taxonomy_verbose(str(raw).strip(), valid_sysids)
                method_list.append(method)
                if method != "exact" and method != "empty":
                    root = str(raw).strip().split("/")[0]
                    collection_issues.setdefault(root, []).append(method)

            method_series = pd.Series(method_list)
            counts = method_series.value_counts()

            st.write("### Method Distribution")
            dist_df = pd.DataFrame({
                "Method": counts.index,
                "Count": counts.values,
                "Percentage": (counts.values / len(method_list) * 100).round(1),
                "Icon": [_METHOD_COLOURS.get(m, "❓") for m in counts.index],
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

            if collection_issues:
                st.write("### Collections with Most Issues")
                coll_rows = []
                for root, methods in sorted(collection_issues.items(), key=lambda x: -len(x[1])):
                    coll_rows.append({
                        "Collection Root": root,
                        "Issue Count": len(methods),
                        "Primary Issue": pd.Series(methods).mode().iloc[0] if methods else "—",
                        "Methods": ", ".join(sorted(set(methods))),
                    })
                st.dataframe(pd.DataFrame(coll_rows[:20]), use_container_width=True, hide_index=True)

            # Exact match rate
            exact_rate = (method_series == "exact").sum() / len(method_series) * 100
            st.metric("Exact Match Rate", f"{exact_rate:.1f}%")
        else:
            st.info("No data to analyse. Add training data or feedback first.")

# ── Tab 5: Knowledge Base Search ──────────────────────────────────
with kb_tab:
    st.markdown("""
    **Reverse lookup**: search by **folder name** or **description** to find the correct SysID.
    Useful when you know *what* the folder is but not *its code*.

    Uses fuzzy matching (substring + close-match scoring) against
    the taxonomy's folder name map.
    """)

    from difflib import get_close_matches

    kb_query = st.text_input("Search by folder name or keyword", "", key="kb_search_input")

    if kb_query.strip():
        q = kb_query.strip().lower()

        # Strategy 1: Substring match (broad)
        substring_hits = {k: v for k, v in folder_name_map.items() if q in (v or "").lower() or q in k.lower()}

        # Strategy 2: Fuzzy match on folder names (for typos in the query itself)
        all_names = list(folder_name_map.values())
        close_names = get_close_matches(q, [n.lower() for n in all_names], n=15, cutoff=0.4)
        fuzzy_hits = {}
        for cn in close_names:
            for k, v in folder_name_map.items():
                if v.lower() == cn and k not in substring_hits:
                    fuzzy_hits[k] = v

        # Strategy 3: Token overlap (for multi-word searches like "rolling stock")
        query_tokens = set(q.split())
        token_hits = {}
        if len(query_tokens) > 1:
            for k, v in folder_name_map.items():
                if k in substring_hits or k in fuzzy_hits:
                    continue
                name_tokens = set((v or "").lower().split())
                overlap = query_tokens & name_tokens
                if len(overlap) >= min(2, len(query_tokens)):
                    token_hits[k] = v

        # Strategy 4: Hierarchical path search (search within SysID structure)
        path_hits = {}
        for k, v in folder_name_map.items():
            if k in substring_hits or k in fuzzy_hits or k in token_hits:
                continue
            # Check if query matches any segment's folder name in the hierarchy
            parts = k.split("/")
            for depth in range(len(parts), 0, -1):
                ancestor = "/".join(parts[:depth])
                ancestor_name = folder_name_map.get(ancestor, "")
                if q in ancestor_name.lower():
                    path_hits[k] = v
                    break

        # Merge results with scoring
        merged = []
        for k, v in substring_hits.items():
            score = "exact" if q == v.lower() else "substring"
            merged.append({"SysID": k, "Folder Name": v, "Match Type": f"🟢 {score}", "Depth": k.count("/")})
        for k, v in fuzzy_hits.items():
            merged.append({"SysID": k, "Folder Name": v, "Match Type": "🟡 fuzzy", "Depth": k.count("/")})
        for k, v in token_hits.items():
            merged.append({"SysID": k, "Folder Name": v, "Match Type": "🔵 token overlap", "Depth": k.count("/")})
        for k, v in path_hits.items():
            merged.append({"SysID": k, "Folder Name": v, "Match Type": "🟣 hierarchy", "Depth": k.count("/")})

        if merged:
            result_df = pd.DataFrame(merged)
            st.write(f"**{len(result_df)} matches** for '{kb_query.strip()}'")
            st.dataframe(result_df, use_container_width=True, height=400)

            # Quick-copy helper
            if len(result_df) > 0:
                st.caption("Click a SysID above to use it in other tabs or forms.")
        else:
            st.warning(f"No matches found for '{kb_query.strip()}'. Try broader keywords or check spelling.")

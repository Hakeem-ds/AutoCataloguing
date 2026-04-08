# Handover Document — Corporate Archives Auto-Classification

> **Author:** Hakeem Fujah  
> **Date:** April 2026  
> **Version:** 3.1  
> **Status:** Production — deployed on Databricks Apps + Streamlit Community Cloud

---

## 1. What This System Does

An ML-powered document classification tool for the TfL Corporate Archives. Given a document's **title** and **description**, it predicts which archival **SysID folder** (out of 251 classes) the document belongs to, with confidence scoring and related-context suggestions.

**Key stats:**
- 82% accuracy on held-out test set
- 251 archival folder classes, 1,707 taxonomy entries
- ~1 minute serverless retrain cycle
- Confidence bands: HIGH (≥92%, 99.7% reliable), MODERATE (23–92%), LOW (<23%)
- Related-context feature recovers ~56% of wrong predictions via suggestions

---

## 2. Architecture Overview

```
                         ┌─────────────────────────────┐
                         │   Databricks Workspace       │
                         │                              │
  User ──→ Streamlit ──→ │  MLflow Tracking Server      │
           Cloud App     │  (experiments, model artifacts)
                         │                              │
                         │  Databricks Job (serverless)  │
                         │  (retrain pipeline, ~1 min)   │
                         └─────────────────────────────┘
```

**Three main components:**
1. **Streamlit App** — UI for prediction, taxonomy management, feedback, model management
2. **Training Pipeline** (`ac_pipe/`) — data prep, TF-IDF+SVM training, evaluation, MLflow logging
3. **MLflow** — experiment tracking, model artifact storage, metric history

**Detailed architecture:** See `ARCHITECTURE.md` (Mermaid diagrams, module graph, file matrix)  
**User guide:** See `Autoclassification Scheme/streamlit_demo/streamlit_app/USER_GUIDE.md`

---

## 3. Deployment Options

### Option A: Streamlit Community Cloud (Current — Recommended)

**URL:** https://autocataloguing.streamlit.app (or your configured subdomain)  
**Cost:** Free  
**Access:** Anyone with the link — no Databricks login needed  

**How it works:**
- GitHub push → Streamlit Cloud auto-deploys
- App reads model artifacts from Databricks MLflow via REST API (PAT auth)
- Taxonomy (`folder_name_map.json`) is bundled in the repo
- Model downloaded and cached on first load (~18 seconds)

**Key files:**
| File | Purpose |
|---|---|
| `streamlit_app.py` | Entry point (repo root, avoids spaces in path) |
| `requirements.txt` | Python dependencies (repo root) |
| `.python-version` | Pins Python 3.10 |
| `pages/*.py` | Page wrappers (set sys.path, exec real pages) |

**Secrets (Streamlit Cloud → Settings → Secrets):**
```toml
DATABRICKS_HOST = "https://adb-2223129477745111.11.azuredatabricks.net"
DATABRICKS_TOKEN = "dapi..."
TRAINING_JOB_ID = "922802212201609"
MLFLOW_TRACKING_URI = "databricks"
MLFLOW_EXPERIMENT = "/Users/hakeemfujah@tfl.gov.uk/experiments/ac_model_v2"
```

**Limitations:**
- Filesystem is ephemeral — feedback.csv resets on redeploy
- Sleeps after 7 days of inactivity (~30s cold start on wake)
- 1 GB memory limit (sufficient for current model)
- Retraining requires Databricks workspace to be online

**To redeploy:** Push to `main` branch on GitHub. Streamlit auto-detects and redeploys.

### Option B: Databricks Apps

**URL:** https://archive-doc-classifier-2223129477745111.11.azure.databricksapps.com  
**Cost:** Included in Databricks workspace  
**Access:** Requires Databricks workspace authentication (Azure AD)

**How it works:**
- Deployed via `databricks.yml` in the streamlit_app directory
- App runs on Databricks-managed compute
- Direct access to MLflow, workspace files, and Databricks APIs
- Feedback persists in workspace filesystem

**Key files:**
| File | Purpose |
|---|---|
| `src/main.py` | Entry point (secrets bridge, sklearn shims, NLTK) |
| `src/app.yaml` | Streamlit server config (XSRF/CORS disabled for uploads) |
| `databricks.yml` | Databricks Apps deployment config |
| `src/requirements.txt` | Python dependencies |

**To redeploy:**
```bash
# From Databricks workspace
databricks apps deploy archive-doc-classifier
```

**Advantages over Streamlit Cloud:**
- Persistent filesystem (feedback, taxonomy edits survive redeploys)
- Direct MLflow access (no PAT needed)
- No sleep/wake cycles
- Azure AD authentication

### Option C: Azure Container Apps (Prepared, Not Deployed)

Deployment scripts are ready in `deploy/` but require Azure resource group creation permissions.
- `deploy/azure-container-apps.sh` — automated 8-step deployment
- `deploy/.env.aca.example` — config template
- `.github/workflows/deploy-aca.yml` — GitHub Actions CI/CD (needs OIDC setup)
- Scale-to-zero, ~£3-8/month

---

## 4. Key Resources

| Resource | Location |
|---|---|
| **GitHub repo** | https://github.com/Hakeem-ds/AutoCataloguing |
| **Streamlit Cloud app** | https://autocataloguing.streamlit.app |
| **Databricks App** | https://archive-doc-classifier-2223129477745111.11.azure.databricksapps.com |
| **MLflow experiment** | `/Users/hakeemfujah@tfl.gov.uk/experiments/ac_model_v2` (ID: 365200970488492) |
| **Retrain job** | Job ID `922802212201609` ("Archive Classifier Retrain v2.1") |
| **Model comparison notebook** | `/Users/hakeemfujah@tfl.gov.uk/Model Comparison TF-IDF vs HF Embeddings` |
| **App source** | `Autoclassification Scheme/streamlit_demo/streamlit_app/src/` |
| **Pipeline source** | `ac_pipe/` |
| **Architecture doc** | `Autoclassification Scheme/streamlit_demo/streamlit_app/ARCHITECTURE.md` |

---

## 5. Day-to-Day Operations

### 5.1 The Model Isn't Loading

1. Go to **🧬 Model Versions** page
2. Click **🔄 Sync from MLflow** to discover trained models
3. Select a model version from the table
4. Click **⭐ Promote this version** to make it active
5. Return to **🔮 Predict** — model is now loaded

> The app auto-syncs on first load. If it fails (timeout, secrets misconfigured), use manual sync.

### 5.2 Retraining the Model

1. Collect feedback via the Predict page (corrections accumulate in `feedback.csv`)
2. Go to **🔁 Review & Retrain**
3. Click **Build Training Data** (merges feedback with existing training set)
4. Click **Trigger Retrain Job** (runs Databricks serverless job, ~1 minute)
5. Go to **🧬 Model Versions** → **🔄 Sync from MLflow** → **⭐ Promote** the new version

### 5.3 Managing the Taxonomy

- **Browse/search:** Taxonomy Manager page, search by SysID or folder name
- **Add entry:** Enter SysID + folder name, enable "Auto-create missing parents"
- **Bulk import:** Upload CSV with `sys_id` and `folder_name` columns
- **Export:** Download as JSON or CSV from the toolbar buttons

### 5.4 Updating the Databricks PAT

PAT tokens expire. To update:
- **Streamlit Cloud:** Settings → Secrets → update `DATABRICKS_TOKEN`
- **Databricks Apps:** Update the app secret in Databricks workspace
- **GitHub (if using ACA):** Settings → Secrets → Actions → update `DATABRICKS_TOKEN`

---

## 6. Configuration Reference

### Environment Variables / Secrets

| Variable | Required | Description |
|---|---|---|
| `DATABRICKS_HOST` | Yes | Workspace URL (e.g. `https://adb-xxx.xx.azuredatabricks.net`) |
| `DATABRICKS_TOKEN` | Yes | Personal Access Token (minimum: MLflow read, Jobs trigger) |
| `TRAINING_JOB_ID` | Yes | Databricks Job ID for retraining (`922802212201609`) |
| `MLFLOW_EXPERIMENT` | Yes | MLflow experiment path |
| `MLFLOW_TRACKING_URI` | Yes | Always `databricks` |

### Confidence Bands (in `core/config.py`)

| Band | Threshold | Accuracy in band | Action |
|---|---|---|---|
| 🔴 LOW | < 23% | ~33% | Review required — check suggestions |
| 🟡 MODERATE | 23–92% | ~87% | Spot-check recommended |
| 🟢 HIGH | ≥ 92% | ~99.7% | Auto-accept safe |

### Model Retention

`CONFIG["retention_max_versions"] = 3` — keeps promoted + previous (rollback) + latest trained.

---

## 7. Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| "No models found" on Predict page | Registry empty on fresh deploy | Model Versions → Sync from MLflow → Promote |
| `pkg_resources` deprecation warning | mlflow-skinny uses deprecated API | Harmless; upgrade mlflow-skinny when available |
| sklearn version mismatch warnings | Model trained on 1.1.1, app has 1.2.2 | Retrain to create model on 1.2.2 |
| `ModuleNotFoundError: 'core'` | Page loaded without sys.path setup | Page wrapper missing or broken; check `pages/*.py` at repo root |
| FileNotFoundError on multi-user | `os.chdir()` in entry point | Ensure `streamlit_app.py` has NO `os.chdir()` calls |
| Taxonomy shows "No entries" | `folder_name_map.json` not in git | Check `.gitignore`: need `artifacts/*` (not `artifacts/`) for exceptions to work |
| Excel upload fails | `openpyxl` not in requirements | Add `openpyxl>=3.1.0` to `requirements.txt` |
| Streamlit Cloud install fails | Path with spaces | Entry point must be at repo root (no spaces in path) |
| PAT expired | Databricks token rotation | Update secret in Streamlit Cloud / Databricks Apps |

### Gitignore Gotcha

Git **cannot re-include files** under a directory ignored with trailing `/` (e.g. `artifacts/`).  
Use `artifacts/*` (content glob) instead, then `!artifacts/specific_file.json` works.  
Both root and streamlit_app `.gitignore` files must use this pattern.

### Deserialization Shims

Two shim modules exist for model unpickling — **do NOT delete**:
- `src/training/train.py` — provides `_select_text()` function (12 lines)
- `src/core/pipeline_utils.py` — provides `select_text_column()` (imported by main.py)

The sklearn compat shim in `streamlit_app.py` maps `base_estimator ↔ estimator` for CalibratedClassifierCV objects pickled on sklearn 1.1.1.

---

## 8. Pending Work & Future Improvements

### High Priority
| Item | Description | Status |
|---|---|---|
| **Human-Correction Replay** | Replay feedback.csv through current vs candidate model to measure improvement | Design ready, not implemented |
| **Feedback persistence on Streamlit Cloud** | Filesystem is ephemeral; feedback.csv resets on redeploy | Consider: cloud storage, database, or Databricks API |
| **PAT rotation strategy** | Tokens expire; need automated renewal or service principal | Manual update in secrets |

### Medium Priority
| Item | Description |
|---|---|
| **Rare-Class F1 / Hierarchical Accuracy / ECE** | New evaluation metrics in `training/evaluation.py` |
| **Top-2 accuracy + misclassification severity** | Easy additions to evaluation |
| **Prediction logging** | CSV-based logging of all predictions for drift detection |
| **Confidence trends + soft alerts** | Track confidence distributions across versions |
| **Weekly retraining schedule** | Automated retrain with data sufficiency gate |

### Low Priority
| Item | Description |
|---|---|
| **Excel Integration** | See `EXCEL_INTEGRATION_PLAN.md` — Power Query calling model endpoint |
| **Flatten streamlit_demo/streamlit_app/ nesting** | Requires app redeployment |
| **Fix stale imports in ac_pipe_tests** | Tests reference removed `ingestion.*` modules |
| **Unity Catalog Model Registry** | Requires UC-enabled cluster; upgrade path documented |
| **UI confidence threshold slider** | Let users adjust confidence bands per session |

---

## 9. File Structure (v3.1)

```
Corporate Archives/
├── streamlit_app.py              # Streamlit Cloud entry point
├── requirements.txt              # Streamlit Cloud dependencies
├── .python-version               # Python 3.10
├── runtime.txt                   # Backup Python version spec
├── .gitignore                    # Root gitignore (artifacts/* not artifacts/)
├── README.md                     # Project overview
├── HANDOVER.md                   # This document
├── EXCEL_INTEGRATION_PLAN.md     # Future Excel/Power Query integration
│
├── pages/                        # Streamlit Cloud page wrappers
│   ├── 1_predict.py              #   (sets sys.path, execs real page)
│   ├── 2_diagnostics.py
│   ├── 3_label_map_manager.py
│   ├── 4_review_and_retrain.py
│   └── 5_model_versions.py
│
├── ac_pipe/                      # ML training pipeline
│   ├── main.py                   #   Pipeline entry point
│   ├── config.yaml               #   Training config
│   ├── training/train.py         #   TF-IDF + SVM training
│   ├── training/evaluation.py    #   Metrics + evaluation
│   ├── utils/label_normalisation.py  # Standalone taxonomy normaliser
│   ├── artifacts/                #   Co-located taxonomy JSON
│   │   └── folder_name_map.json
│   └── outputs/                  #   (gitignored) Training outputs
│       └── used_training_data.csv
│
├── Autoclassification Scheme/
│   └── streamlit_demo/
│       └── streamlit_app/
│           ├── Dockerfile         # Multi-stage Docker build
│           ├── databricks.yml     # Databricks Apps config
│           ├── ARCHITECTURE.md    # System architecture reference
│           ├── USER_GUIDE.md      # End-user guide
│           ├── .gitignore         # App-level gitignore (artifacts/*)
│           └── src/
│               ├── main.py        # App entry (Databricks Apps)
│               ├── app.py         # Streamlit home page + How To
│               ├── app.yaml       # Streamlit server config
│               ├── requirements.txt
│               ├── core/          # 13 core modules
│               ├── pages/         # 5 Streamlit pages
│               ├── artifacts/     # Runtime data (tracked in git)
│               │   ├── folder_name_map.json  (1707 entries)
│               │   └── model_registry.json
│               ├── training/      # In-app training shim
│               │   └── train.py   (deserialization shim — DO NOT DELETE)
│               └── assets/        # CSS, logo
│
├── deploy/                       # Azure Container Apps (prepared)
│   ├── azure-container-apps.sh
│   └── .env.aca.example
│
└── .github/workflows/
    └── deploy-aca.yml            # GitHub Actions CI/CD (OIDC auth)
```

---

## 10. Technical Decisions & Rationale

| Decision | Rationale |
|---|---|
| **TF-IDF + SVM** over deep learning | Small dataset (4228 samples), 251 classes. SVM achieves 82% accuracy with ~1 min train time. Transformers would need significant data augmentation. |
| **Custom JSON registry** over MLflow Model Registry | App is file-based, runs on Databricks Apps (no UC). Custom JSON gives full control over promotion, retention, metadata. MLflow Registry is additive for UI visibility. |
| **Serverless compute** for retraining | 9.7× faster than provisioned cluster (1 min vs 10 min). 118× faster startup. Cost-effective for infrequent retrains. |
| **Streamlit Cloud** over Azure Container Apps | No Azure resource group permissions needed. Free. Auto-deploys from GitHub. Trade-off: ephemeral filesystem. |
| **mlflow-skinny** on Streamlit Cloud | Full mlflow is ~300MB with heavy deps. mlflow-skinny is ~10MB. Model loaded via cloudpickle fallback when `mlflow.sklearn` unavailable. |
| **Page wrappers** at repo root | Streamlit multi-page feature looks for `pages/` relative to the main script. Real pages are in `src/pages/`. Wrappers set sys.path and exec the real page. |
| **`artifacts/*` not `artifacts/`** in gitignore | Git cannot re-include files under a directory ignored with `/`. Content glob `*` allows file-level exceptions. |

---

## 11. Contacts & Access

| Item | Details |
|---|---|
| **Developer** | Hakeem Fujah (hakeemfujah@tfl.gov.uk) |
| **GitHub account** | Hakeem-ds |
| **Databricks workspace** | `adb-2223129477745111.11.azuredatabricks.net` |
| **Git user** | `Hakeem-ds` / `hakeemfujah@tfl.gov.uk` |
| **Git push** | Cell 47 in Model Comparison notebook (uses getpass for PAT) |

> **Important:** Update the `DATABRICKS_TOKEN` in Streamlit Cloud secrets if the PAT expires.  
> **Important:** Update the Git push cell (cell 47) if the GitHub account changes.

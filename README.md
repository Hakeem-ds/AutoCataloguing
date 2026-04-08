# Corporate Archives — Document Auto-Classification

An ML-powered document classification system for the TfL Corporate Archives.
Predicts archival folder placement (251 classes) from document title and description,
with **related context** guidance to help reviewers correct predictions.

**Live app:** [autocataloguing.streamlit.app](https://autocataloguing.streamlit.app)

## Key Features

| Feature | Description |
|---|---|
| **Single & Batch Prediction** | Enter text or upload CSV/Excel files |
| **Confidence Bands** | 🟢 HIGH (≥92%), 🟡 MODERATE (23–92%), 🔴 LOW (<23%) |
| **Related Context** | Top folders, neighbourhood suggestions, sibling classes |
| **Feedback Loop** | Correct → Retrain → Promote (full ML lifecycle) |
| **Taxonomy Manager** | 1,707 entries, bulk import, audit trail, aliases |
| **Model Versioning** | MLflow-backed, side-by-side metrics, promote/rollback |
| **Serverless Retrain** | ~1 minute via Databricks serverless compute |

## Model

| Property | Value |
|---|---|
| Architecture | TF-IDF + Linear SVM (CalibratedClassifierCV) |
| Accuracy | 82.3% on held-out test |
| Classes | 251 archival folders |
| Training samples | 4,228 (cumulative with feedback) |
| Retrain time | ~1 minute (serverless) |
| Confidence gap | 0.616 (correct vs wrong mean) |

## Deployment

| Platform | URL | Access | Cost |
|---|---|---|---|
| **Streamlit Cloud** | [autocataloguing.streamlit.app](https://autocataloguing.streamlit.app) | Public link | Free |
| **Databricks Apps** | Workspace-internal | Azure AD login | Included |
| **Azure Container Apps** | Scripts ready in `deploy/` | Public link | ~£3-8/mo |

## Project Structure

```
Corporate Archives/
├── streamlit_app.py              # Streamlit Cloud entry point
├── requirements.txt              # Dependencies
├── .python-version               # Python 3.10
├── pages/                        # Streamlit Cloud page wrappers
├── HANDOVER.md                   # Handover documentation
├── EXCEL_INTEGRATION_PLAN.md     # Future Excel integration
│
├── ac_pipe/                      # ML training pipeline
│   ├── main.py                   #   Entry point (serverless job)
│   ├── training/                 #   TF-IDF + SVM training
│   ├── utils/                    #   Standalone label normalisation
│   └── artifacts/                #   Co-located taxonomy JSON
│
├── Autoclassification Scheme/
│   └── streamlit_demo/
│       └── streamlit_app/
│           ├── Dockerfile
│           ├── ARCHITECTURE.md   # System architecture (Mermaid diagrams)
│           ├── USER_GUIDE.md     # End-user guide
│           └── src/              # App source (core/, pages/, artifacts/)
│
└── deploy/                       # Azure Container Apps scripts
```

## Quick Start

### For Users
1. Open [autocataloguing.streamlit.app](https://autocataloguing.streamlit.app)
2. Go to **🔮 Predict** → enter title + description → get folder prediction
3. For batch: upload a CSV with `title` and `description` columns

### For Developers
1. Clone: `git clone https://github.com/Hakeem-ds/AutoCataloguing.git`
2. Set secrets in Streamlit Cloud (see HANDOVER.md §3)
3. Push to `main` → auto-deploys

### Model Not Loading?
1. Go to **🧬 Model Versions** → click **🔄 Sync from MLflow**
2. Select a version → click **⭐ Promote**
3. Return to Predict — model is now active

## Feedback Loop

```
Predict → Review → Correct → Build Training Data → Retrain (serverless) → New Version → Promote
```

## Tech Stack

- **ML:** scikit-learn (TF-IDF + LinearSVC), MLflow, Databricks Jobs (serverless)
- **App:** Streamlit (Databricks Apps + Community Cloud)
- **Infra:** Azure Databricks, GitHub, Streamlit Cloud
- **Model loading:** mlflow-skinny + cloudpickle fallback for Streamlit Cloud

## Documentation

| Document | Description |
|---|---|
| `HANDOVER.md` | Operational handover (deployment, troubleshooting, pending work) |
| `ARCHITECTURE.md` | System architecture, module graph, file matrix, debugging reference |
| `USER_GUIDE.md` | End-user guide for the Streamlit app |
| `EXCEL_INTEGRATION_PLAN.md` | Future Excel/Power Query integration plan |

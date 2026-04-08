# Excel Integration Plan — Archive Document Classifier

> **Status:** Future improvement — replacing the Streamlit app with a lightweight Excel-based workflow.  
> **Date:** April 2026  
> **Author:** Hakeem Fujah  

---

## Motivation

The Streamlit app has served well for prototyping and internal demos, but for production adoption:

- Users already work in Excel — adding a new browser app creates friction  
- The app requires Databricks Apps compute, Docker configuration, and ongoing UI maintenance  
- Excel integration lets users predict directly in their existing workflow  
- IT approval is simpler (no new URLs, no new apps)  

---

## Architecture Overview

```
┌─────────────────┐     REST API      ┌──────────────────────────┐
│   Excel Workbook │ ───────────────► │  Databricks Model Serving │
│  (Power Query /  │ ◄─────────────── │  Endpoint                 │
│   Office Scripts)│   JSON response  │  (MLflow pyfunc model)    │
└────────┬────────┘                   └──────────────────────────┘
         │                                        ▲
         │ Corrections                            │ Auto-promote
         ▼                                        │
┌─────────────────┐                   ┌──────────────────────────┐
│  Delta Table     │ ◄─────────────── │  Databricks Job           │
│  feedback table  │   MERGE          │  (ac_pipe retraining)     │
└─────────────────┘                   └──────────────────────────┘
```

---

## Phase 1: Model Serving Endpoint

### Step 1.1 — Register model in Unity Catalog

Promote the MLflow model from `runs:/` URI to a registered model in Unity Catalog:

```
catalog.schema.ac_classifier
```

This gives versioning, lineage, and access control via UC.

### Step 1.2 — Create Model Serving endpoint

Create a Databricks Model Serving endpoint that:

- Accepts: `{"Title": "...", "Description": "..."}`  
- Returns: `{"SysID": "...", "Folder": "...", "Confidence": 0.94, "Band": "HIGH"}`  
- Scales to zero when idle (cost control)  
- Auto-scales on demand  

### Step 1.3 — Custom inference wrapper

Wrap the sklearn pipeline in a custom `mlflow.pyfunc.PythonModel` that includes:

- Label normalisation (`normalise_to_taxonomy`)  
- Folder name resolution (`_resolve_folder_name`)  
- Confidence extraction and banding  
- Top-5 related suggestions  

This ensures the endpoint returns the same rich output as the Streamlit app.

### Step 1.4 — Authentication

Generate a Personal Access Token (PAT) or configure OAuth/service principal for Excel to authenticate against the endpoint.

---

## Phase 2: Excel Integration

### Step 2.1 — Power Query connector (recommended)

Custom M function that calls the REST endpoint. Works in Excel Desktop and Excel Web.

**User workflow:**
1. Paste Title and Description into columns A and B  
2. Refresh the Power Query table  
3. Predicted SysID, Folder, Confidence appear in columns C-E  

**Pros:** No macros, no VBA, works on Mac/Windows/Web, refreshable.

### Step 2.2 — Office Scripts (alternative)

TypeScript function callable from Excel Online via the Automate tab.

```typescript
async function predict(title: string, description: string): Promise<string> {
    const response = await fetch(ENDPOINT_URL, {
        method: "POST",
        headers: { "Authorization": "Bearer " + PAT },
        body: JSON.stringify({ "dataframe_records": [{ "Title": title, "Description": description }] })
    });
    const result = await response.json();
    return result.predictions[0].SysID;
}
```

**Pros:** Interactive, button-triggered. **Cons:** Excel Online only.

### Step 2.3 — VBA macro (legacy fallback)

`XMLHTTP` call to the endpoint for Excel Desktop users without Power Query.

**Use only if** Power Query is unavailable in the organisation.

### Step 2.4 — Batch mode

Power Query reads a named table of rows, sends them as a batch to the endpoint, and populates SysID + Folder + Confidence columns. Supports hundreds of rows per refresh.

---

## Phase 3: Feedback Collection via Excel

### Step 3.1 — Correction column in workbook

User fills a `Correct SysID` column next to predictions. A "Submit Corrections" button (Office Script or Power Automate flow) sends rows where `Correct != Predicted` to a Delta table:

```sql
catalog.schema.feedback (
    Title STRING,
    Description STRING,
    Predicted_SysID STRING,
    Correct_SysID STRING,
    Model_Version STRING,
    Timestamp TIMESTAMP,
    Submitted_By STRING
)
```

### Step 3.2 — SharePoint/OneDrive alternative

Corrections saved to a shared Excel file on SharePoint. A Databricks workflow picks it up on schedule via the Microsoft Graph API or a mounted ADLS path.

### Step 3.3 — Dedup on ingestion

Same `Title + Description + Correct SysID` dedup logic, implemented as a SQL MERGE:

```sql
MERGE INTO catalog.schema.feedback AS target
USING new_corrections AS source
ON target._dedup_key = source._dedup_key
WHEN NOT MATCHED THEN INSERT (...)
```

---

## Phase 4: Automated Retraining

### Step 4.1 — Scheduled workflow

Existing Databricks Job (`Archive Classifier Retrain v2.1`, ID `922802212201609`) triggered:

- On a schedule (e.g., weekly), OR  
- When correction count exceeds the dynamic threshold  

### Step 4.2 — Read from Delta table

Modify `ac_pipe/main.py` to accept a third data source mode:

```
python main.py --feedback_table catalog.schema.feedback
```

Reads corrections from the Delta table instead of CSV artifacts. Cumulative merge with `used_training_data` remains unchanged.

### Step 4.3 — Auto-promote and endpoint update

After retraining:

1. Register new model version in Unity Catalog  
2. Update the Model Serving endpoint to route traffic to the new version  
3. Zero-downtime via canary/traffic splitting (e.g., 90% old / 10% new, then full cutover)  

### Step 4.4 — Notifications

Databricks job notification on completion — email or Microsoft Teams webhook. User knows when Excel predictions will use the updated model.

---

## Architecture Comparison

| Aspect | Streamlit App (current) | Excel + Endpoint (proposed) |
| --- | --- | --- |
| **Deployment** | Databricks Apps + Docker | Model Serving + Excel workbook |
| **Maintenance** | App code, UI bugs, dependencies | Endpoint config only |
| **User access** | Browser link | Existing Excel workflow |
| **Cost** | Apps compute + endpoint | Endpoint compute only (scales to zero) |
| **Retraining** | In-app button | Scheduled or threshold-triggered |
| **Offline use** | No | Partial (cached predictions) |
| **IT approval** | New app URL needed | Excel already approved |

---

## Implementation Order

| Priority | Step |
| --- | --- | 
| 1 | 1.1 Register model in UC 
| 2 | 1.2 Create serving endpoint 
| 3 | 1.3 Custom pyfunc wrapper 
| 4 | 2.1 Power Query connector 
| 5 | 3.1 Correction column + submit flow 
| 6 | 4.1 Scheduled retraining workflow 
| 7 | 4.3 Auto-promote + endpoint update

**Total estimated effort:** ~16 hours (2 working days)

---

## Dependencies

- Databricks Model Serving enabled on workspace  
- Unity Catalog access for model registration  
- Excel 2016+ or Excel Online (for Power Query / Office Scripts)  
- Network connectivity from user machines to Databricks endpoint  

---

## Migration Path

1. Deploy endpoint alongside existing Streamlit app (both live)  
2. Distribute Excel workbook template to pilot users  
3. Collect feedback via Excel for 2-4 weeks  
4. Once validated, decommission Streamlit app  
5. Streamlit code preserved in GitHub repo as reference implementation

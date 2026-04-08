# Corporate Archives Auto-Classification — User Guide

> **Who is this for?** Anyone using the app to classify documents into archival
> folders. No technical knowledge required.
>
> **What does the app do?** You give it a document’s title and description, and
> it predicts which archival folder (SysID) the document belongs to. It also
> shows you related folders and suggestions so you can quickly correct any
> mistakes.

---

## Getting Started

Open the app in your browser. You’ll see a sidebar on the left and the main
content area on the right.

The **sidebar** contains:
- **Settings** — controls how sensitive the suggestions are (more on this below)
- **Page navigation** — five pages listed vertically

The **main area** shows whichever page you’re on. By default, you’ll land on
the **Predict** page.

---

## Page 1: Predict

This is the page you’ll use most. It has three sections.

### Single Prediction

**Step 1 — Enter the document details**

On the left side you’ll see two fields:
- **Title** — the document’s title (e.g. “Board Meeting Minutes, 14 June 1985”)
- **Description** — a brief description of the document’s content

Fill in at least one of these. The more detail you provide, the better the
prediction will be.

**Step 2 — Click “Predict”**

The app will analyse the text and show you a result card on the right:

- **SysID** — the predicted folder reference (e.g. `LT000266/004/003`)
- **Folder** — the human-readable folder name (e.g. “Station Photography”)
- **Confidence** — how certain the model is, shown as a percentage and a
  colour-coded label:

| Colour | Label | What it means |
|--------|-------|---------------|
| Green | **High** | The model is very confident. The prediction is almost certainly correct. |
| Orange | **Moderate** | The model is fairly confident, but you should double-check. Suggestions are shown to help. |
| Red | **Low** | The model is uncertain. Please review the suggestions carefully — the correct folder is likely among them. |

**Step 3 — Review the Related Context**

Below the prediction card, you’ll see a **Related Context** panel. This is
your guide for reviewing the prediction:

- **Related Folders** — a table showing the next most likely folders, with
  their SysID, name, how they relate to the prediction (sibling, same branch,
  etc.), and their own confidence score.

- **Siblings** — other folders that sit under the same parent. These appear as
  small tags. If the prediction is in the right area but the wrong specific
  folder, the correct one is probably here.

- **Suggested Neighbourhoods** (orange and red predictions only) — when the
  model is less certain, the app groups related folders by their parent and
  shows you the broader area where the correct folder is likely to be. Each
  neighbourhood shows:
  - The **parent folder** name and SysID
  - The **child folders** within it (with their SysIDs)
  - A **combined confidence** percentage

  Think of it like this: even if the model can’t pinpoint the exact folder, it
  can often identify the right *neighbourhood*. This helps you narrow down your
  search.

**Step 4 — Submit a correction (if needed)**

If the prediction is wrong, scroll down to **Suggest a Correction**.

You have two options:
1. **Pick from the dropdown** — the app pre-fills this with all the related
   folders and neighbourhood suggestions it showed you. Just select the correct
   one.
2. **Type a custom SysID** — if the correct folder isn’t in the list, choose
   “-- Type a custom SysID --” and enter it manually.

Click **Submit Correction**. The app will save your correction for future model
improvement.

---

### Batch Prediction

Use this when you have many documents to classify at once.

**Step A — Upload your file**

Prepare a CSV or Excel file with at least two columns: one for titles and one
for descriptions. Upload it using the file uploader.

You’ll see a preview of your data. Then select which column is the **Title**
and which is the **Description**.

Optionally, if your file already has a column with the correct folder labels
(e.g. from a previous review), select it as the **Correct Label Column**.

**Step B — Run prediction**

Click **Predict Entire File**. The app will process every row and show a
results table with:

| Column | What it shows |
|--------|---------------|
| Title | The document title |
| Predicted SysID | The model’s predicted folder |
| Predicted Folder | Human-readable folder name |
| Confidence | How certain the model is |
| Band | HIGH, MODERATE, or LOW |
| Related Context | A summary of nearby folders |

You can **download** the results as a CSV file. This is useful for offline
review — add your corrections to the “Correct SysID” column and re-upload.

**Step C — Ingest corrections**

If your file had a Correct Label Column, click **Ingest Corrections From This
File** to feed them back into the system.

---

### Corrections Only

Already have a corrected file from a previous batch run? Upload it here
without re-running predictions. Just map the columns and click **Ingest
Corrections**.

---

## Adjusting Confidence Settings

In the **sidebar**, under **Settings**, you’ll find a dropdown called
**Confidence threshold**. This controls when the app shows neighbourhood
suggestions:

| Preset | When suggestions appear | Best for |
|--------|------------------------|----------|
| **Standard (calibrated)** | Below 92% confidence | Normal use — balanced accuracy and helpfulness |
| **Conservative (more suggestions)** | Below 95% confidence | When you want extra guidance on more predictions |
| **Permissive (fewer suggestions)** | Below 85% confidence | When you trust the model and want a cleaner view |

The current thresholds are shown beneath the dropdown as a summary line
(e.g. “LOW < 23% • MODERATE 23–95% • HIGH ≥ 95%”).

---

## Page 2: Diagnostics

This page helps you understand how well the model is performing.

- **Label Audit** — shows which SysIDs the model has seen in training and
  which ones are new (unseen). If a folder’s SysID has never appeared in the
  training data, the model cannot predict it.
- **Evaluation** — upload a test file with known correct labels and the app
  will show you accuracy metrics.

You generally won’t need this page for day-to-day use. It’s helpful when
assessing whether the model needs retraining.

---

## Page 3: Label Map Manager

This page manages the mapping between SysIDs (e.g. `LT000266/004/003`) and
human-readable folder names (e.g. “Station Photography”).

Use it to:
- **Search** for a SysID and see its current name
- **Add or update** folder names
- **Download** the full mapping as CSV or JSON
- **Upload** a revised mapping file

Changes here are tracked in an audit log.

---

## Page 4: Review & Retrain

This page is for model improvement. It brings together feedback (corrections)
and training data.

- **Review feedback** — see all corrections that have been submitted
- **Build training data** — merge corrections into the training set
- **Trigger retraining** — start a new training run (this creates a new model
  version that can be compared and promoted on the Model Versions page)

Retraining is typically done periodically (e.g. weekly or after a batch of
corrections), not after every single correction.

---

## Page 5: Model Versions

This page manages which model version the app uses.

- **Sync** — discovers new model versions from recent training runs
- **Compare** — see accuracy metrics side by side
- **Promote** — set a new version as the active production model
- **Smoke test** — run a quick test to verify the new version works

The app always uses the **promoted** version for predictions. When you promote
a new version, the previous one is kept as a rollback safety net.

---

## Tips for Best Results

1. **Be descriptive.** The more detail in the title and description, the
   better the prediction. “Minutes” alone is vague; “Board Meeting Minutes,
   Finance Committee, Q3 2019” gives the model much more to work with.

2. **Trust the suggestions.** When the model shows orange or red confidence,
   the correct folder is usually in the neighbourhood suggestions. Check those
   first before searching manually.

3. **Always submit corrections.** Even if you found the right folder yourself,
   submitting the correction helps the model learn. Over time, the model gets
   better at the folders you correct most often.

4. **Use batch mode for large sets.** If you have more than a handful of
   documents, upload them as a CSV. It’s faster and you can review and correct
   offline.

5. **Check the confidence column in batch results.** Sort by the Band column
   to focus your review time on LOW and MODERATE predictions — the HIGH ones
   are almost always correct.

---

## Frequently Asked Questions

**Q: The folder name shows as a raw code (e.g. LT000266/004/003) instead of a
readable name. Why?**

The folder isn’t in the name mapping yet. Go to the **Label Map Manager** page
and add a name for that SysID.

**Q: I submitted a correction but the model still predicts the wrong folder.
Why?**

Corrections are stored as feedback but don’t change the model immediately.
The model needs to be **retrained** (Page 4) and the new version **promoted**
(Page 5) before corrections take effect.

**Q: Can I undo a correction?**

Not directly from the app. The feedback file can be edited manually if needed
(ask your administrator).

**Q: The app says “No model versions found.” What do I do?**

Go to the **Model Versions** page and click **Sync**. This discovers trained
models and registers them. If no models appear after syncing, a model needs to
be trained first (Page 4).

**Q: What does “Suggested Neighbourhoods” mean?**

When the model is unsure, it groups related folders by their parent area. For
example, if it’s not sure which specific photography folder a document belongs
to, it might show you the “Photographs Collection” neighbourhood with all its
child folders listed. The correct folder is very likely somewhere in that
neighbourhood.

**Q: How accurate is the model?**

Overall accuracy is about 82%. For predictions marked **High confidence**, the
model is correct 99.7% of the time. For **Moderate**, about 87%. For **Low**,
about 33% — but the neighbourhood suggestions recover the correct area in most
cases.

**Q: How often should we retrain?**

After accumulating a meaningful batch of corrections (roughly 50–100+). Weekly
is a good cadence if corrections are coming in regularly.

**Q: I'm using the Streamlit Cloud version. Will my feedback persist?**

The Streamlit Cloud filesystem is **ephemeral** — feedback.csv resets when the app
redeploys. For permanent feedback, use the Databricks Apps version, or export your
corrections before a redeploy.

**Q: The app shows a "cold start" spinner. Is it broken?**

No. The free Streamlit Cloud tier sleeps after 7 days of inactivity. The first
visitor triggers a ~30-second startup (installs packages, downloads model). After
that it stays awake as long as there's traffic.

**Q: Can I retrain from Streamlit Cloud?**

Yes. Retraining triggers a **Databricks serverless job** via the API. It takes
about 1 minute. Your Databricks workspace must be online and the PAT must be valid.

---

## Quick Reference

| Task | Where to go |
|------|------------|
| Classify a single document | Page 1 → Single Prediction |
| Classify many documents at once | Page 1 → Batch Prediction |
| Submit a correction | Page 1 → Suggest a Correction |
| Upload corrected results | Page 1 → Corrections Only |
| Check model performance | Page 2 → Diagnostics |
| Add a folder name | Page 3 → Label Map Manager |
| Retrain the model | Page 4 → Review & Retrain |
| Switch to a new model version | Page 5 → Model Versions |
| Adjust suggestion sensitivity | Sidebar → Confidence threshold |

---

*Last updated: 8 April 2026 — v3.1 (Streamlit Cloud deployment, serverless retrain).*

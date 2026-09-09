# Data Engineering

Owner: **Axelle Chandra (24/533796/PA/22614)**.  
Everything about the dataset lives here, from the raw download to the Milestone 2 deliverables and cleaning pipelines.

---

## Layout

| Path | Contents | Committed |
|---|---|---|
| `raw/` | The untouched download, exactly as acquired (`diabetes.csv`) | No |
| `interim/` | Intermediate steps of the cleaning pipeline | No |
| `processed/` | Model-ready output (`dataset_m2_v1.csv`) | Only `dataset_m2_*.csv` |
| `src/` | Modular ingestion, validation, cleaning, and EDA scripts | Yes |
| `notebooks/` | Interactive analysis & lecturer lab notebooks | Yes, outputs stripped |
| `reports/` | Milestone 2 reports, logs, dictionaries, and figures | Yes |
| `schema.yaml` | Feature contract, read dynamically by `web-dashboard/` | Yes |
| `dataset.yaml` | Active dataset metadata, provenance, and frozen pointers | Yes |

**The raw file is immutable.** Nothing writes back into `raw/`. Every transformation reads from `raw/` and writes to `interim/` or `processed/`, which makes the before-and-after evidence in the quality report completely verifiable and reproducible.

---

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r data-engineering/requirements.txt
```

### 2. Run the End-to-End Cleaning & Profiling Pipeline
```bash
# Execute with default settings (Outcome-stratified median imputation & winsorization)
python data-engineering/src/pipeline.py

# Alternatively, specify custom imputation strategy (e.g., KNN imputer)
python data-engineering/src/pipeline.py --impute-strategy knn --outlier-method winsorize
```

---

## Milestone 2 Deliverables Checklist

| Item | Where | Status / Summary |
|---|---|---|
| Lab notebook | `notebooks/02_data_engineering_lab.ipynb` | Top-to-bottom executable on fresh kernel with visual outputs |
| Acquisition log | `reports/acquisition-log.md` | 5 concrete entries: source, owner, date, scope, licence, raw location |
| Cleaning log | `reports/cleaning-log.md` | 5 traceable rules detailing zero-to-NaN conversions and imputation |
| Dataset Report | `reports/dataset-report.md` | Source, scope, schema summary, versioning, and clinical limitations |
| Data Dictionary | `reports/data-dictionary.md` | Matches `schema.yaml` field for field |
| Data Quality Report | `reports/data-quality-report.md` | Missingness, duplicates, invalid values, outliers, and figures |
| Data Profiling Report | `reports/data-profiling-report.md` | Distributions, class balance, correlations, and 3 modeling plan changes |
| Frozen dataset | `processed/dataset_m2_v1.csv` | Active in `dataset.yaml`, 100% complete ($768 \times 9$) |
| Concept answers | `reports/concept-answers.md` | Two chapter questions (100–150 words each) |

---

## The Contract Between Folders

1. **`schema.yaml`** is what the dashboard builds its input form and dummy data from.
2. **`processed/dataset_m2_v1.csv`** is what the dashboard switches to automatically. The dashboard immediately detects this file and switches from placeholder dummy rows to the real milestone dataset.

# Dataset Acquisition Log

This log records the provenance, environmental context, and validation steps for the raw dataset acquired for Milestone 2 of the AI Model Engineering project.

---

## Acquisition Metadata

| Field | Record |
| :--- | :--- |
| **Dataset Name** | Pima Indians Diabetes Database (Unprocessed) |
| **Source Origin** | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| **Source Repository** | https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset |
| **Acquisition Date** | 2026-08-30 |
| **Data Custodian / Owner** | Axelle Chandra (24/533796/PA/22614) - Data Engineer |
| **Target Population Scope** | Female patients of Pima Indian descent aged $\ge 21$ years residing near Phoenix, Arizona |
| **License / Terms** | Open dataset for research and educational purposes (Public Domain / NIDDK) |
| **Raw Storage Location** | `data-engineering/raw/diabetes.csv` |
| **Raw File Checksum (SHA-256)** | `b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac` |
| **File Format & Dimensions** | CSV, 768 rows $\times$ 9 columns, 23,873 bytes |

---

## Log Entries

### Entry 1: Source Identification and Authenticity Verification
* **Date**: 2026-08-30
* **Action**: Evaluated candidate repositories hosting the original NIDDK Pima Indians Diabetes Dataset.
* **Verification**: Confirmed the unprocessed version against the classic UCI Machine Learning Repository distribution (768 records, 8 clinical/demographic predictor features, 1 binary target).

### Entry 2: Secure Ingestion and SHA-256 Hashing
* **Date**: 2026-08-30
* **Action**: Downloaded `diabetes.csv` directly from the primary Git source into `data-engineering/raw/diabetes.csv`.
* **Verification**: Generated SHA-256 cryptographic digest `b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac` to ensure immutable integrity.

### Entry 3: Schema Conformance Check
* **Date**: 2026-08-30
* **Action**: Executed `validate_raw_schema()` via `src/data_loader.py`.
* **Outcome**: Verified that all 9 expected headers (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, `Outcome`) exist with zero column mutations.

### Entry 4: Population Invariance & Scope Audit
* **Date**: 2026-08-30
* **Action**: Inspected demographics and inclusion criteria.
* **Outcome**: Confirmed all records represent females aged 21 to 81. Noted demographic restriction (Pima population) as a key generalization limitation for downstream healthcare screening.

### Entry 5: Data Version Freezing Policy
* **Date**: 2026-08-30
* **Action**: Configured `.gitignore` to enforce immutability: `data-engineering/raw/**` is excluded from git commits to respect data policy, while `dataset_m2_v1.csv` is tracked under `data-engineering/processed/`.

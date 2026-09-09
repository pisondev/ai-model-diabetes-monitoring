# Milestone 2: Dataset Report

**Project**: Hybrid Diabetes Classification for Healthcare Risk Screening  
**Component**: Data Engineering & Ingestion  
**Data Engineer**: Axelle Chandra (24/533796/PA/22614)  
**Dataset Version**: `v1` (Frozen Milestone 2 artifact: `dataset_m2_v1.csv`)

---

## 1. Dataset Source & Scope

The dataset utilized in this project is the **Pima Indians Diabetes Dataset**, originally curated by the **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)**.

* **Sample Unit**: One female patient record of Pima Indian heritage, aged $\ge 21$ years.
* **Volume**: 768 total records.
* **Predictor Features**: 8 baseline clinical and physiological variables.
* **Target Feature**: Binary classification label (`Outcome`), where $1$ denotes diabetes diagnosis and $0$ denotes non-diabetic.

---

## 2. Schema Summary

| Feature | Type | Valid Range / Unit | Missing Policy |
| :--- | :--- | :--- | :--- |
| `Pregnancies` | Integer | $0 - 17$ (count) | Zero is valid observation |
| `Glucose` | Float | $44.0 - 200.0\ \text{mg/dL}$ | $0$ marked as NaN and imputed |
| `BloodPressure` | Float | $24.0 - 122.0\ \text{mm Hg}$ | $0$ marked as NaN and imputed |
| `SkinThickness` | Float | $7.0 - 99.0\ \text{mm}$ | $0$ marked as NaN and imputed |
| `Insulin` | Float | $14.0 - 846.0\ \mu\text{U/ml}$ | $0$ marked as NaN and imputed |
| `BMI` | Float | $18.0 - 67.1\ \text{kg/m}^2$ | $0$ marked as NaN and imputed |
| `DiabetesPedigreeFunction` | Float | $0.07 - 2.42$ (score) | Continuous family risk metric |
| `Age` | Integer | $21 - 81$ (years) | Demographic age |
| `Outcome` (Target) | Binary | $\{0, 1\}$ | Prevalence: 34.9% positive |

---

## 3. Dataset Versions

* **Raw Download (`v0`)**: `data-engineering/raw/diabetes.csv` (SHA-256: `b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac`). Untouched and uncommitted.
* **Cleaned & Imputed (`dataset_m2_v1.csv`)**: $768 \times 9$, zero NaNs, winsorized, baseline features formatted for modeling and dashboard consumption.
* **Engineered Feature Set (`dataset_m2_v1_engineered.csv`)**: $768 \times 15$, includes WHO BMI categories, OGTT risk tiers, age bins, pregnancy rate, and HOMA-IR proxy.

---

## 4. Key Limitations

1. **Cohort Homogeneity**: All participants are adult females of Pima Indian heritage. Models trained on this dataset will exhibit demographic bias and should not be deployed across diverse multi-ethnic populations without external recalibration.
2. **Definitional Target Relationship**: Glucose and OGTT metrics are strongly correlated with diabetes status. Evaluation protocols must ensure transparency regarding feature attribution.
3. **Clinical Application Scope**: The system serves as an academic screening prototype and does not substitute for formal clinical diagnostics.

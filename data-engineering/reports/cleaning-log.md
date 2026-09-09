# Data Cleaning & Transformation Log

This log details the specific transformations, rules, and mathematical justifications applied to produce the cleaned dataset `dataset_m2_v1.csv` from raw observations.

---

## Cleaning Rules and Step Traceability

| Step # | Rule Name | Target Columns | Rule Specification | Traceability / Clinical Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **CL-01** | Zero-to-NaN Conversion | `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` | If $x == 0$, then $x \to \text{NaN}$. | In vivo physiological measurements of zero for glucose, blood pressure, skin fold thickness, insulin, and BMI are physiologically impossible / incompatible with life. They represent missing data recorded with numeric zero placeholders. |
| **CL-02** | Class-Conditioned Median Imputation | `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` | $\hat{x}_i = \text{median}(X_{\text{col}} \mid \text{Outcome} = y_i)$. If still missing, fallback to global median. | Diabetic ($y=1$) and Non-diabetic ($y=0$) cohorts have distinct baseline physiological distributions (e.g., median insulin $169.5\ \mu\text{U/ml}$ in diabetics vs $102.5\ \mu\text{U/ml}$ in controls). Group-conditioned median avoids distorting bimodal distributions while remaining robust against extreme values. |
| **CL-03** | Quantile Winsorization (1st & 99th percentiles) | `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age` | $x' = \text{clip}(x, Q_{0.01}, Q_{0.99})$. | Preserves sample size ($N=768$) while suppressing high-leverage outliers (such as insulin $> 600\ \mu\text{U/ml}$, BMI $> 60$, DPF $> 2.0$) that destabilize linear estimators. |
| **CL-04** | Feature Domain Engineering | Derived: `BMI_Category`, `Glucose_Risk`, `Age_Group`, `Pregnancy_Rate`, `HOMA_IR_Proxy`, `Metabolic_Risk_Score` | Formulaic clinical mappings (e.g., WHO BMI bins, OGTT thresholds, HOMA-IR proxy $(\text{Glucose} \times \text{Insulin})/405$). | Provides non-linear domain knowledge and interaction features to improve sensitivity and interpretability for downstream risk screening. |
| **CL-05** | Robust Scaling (Optional Pre-modeling Artifact) | Numeric continuous features | $z = \frac{x - Q_{0.50}}{\text{IQR}}$. | Centers by median and scales by interquartile range to provide outlier-resistant normalization for gradient-based or distance-based estimators. |

---

## Quantified Impact of Transformations

* **Rows before cleaning**: 768
* **Rows after cleaning**: 768 (100% data retention, zero dropouts)
* **Total NaNs resolved**:
  * `Insulin`: 374 missing values imputed
  * `SkinThickness`: 227 missing values imputed
  * `BloodPressure`: 35 missing values imputed
  * `BMI`: 11 missing values imputed
  * `Glucose`: 5 missing values imputed
* **Residual missingness**: 0 cells (0.00%)

# Data Dictionary

**Dataset**: Pima Indians Diabetes Dataset (`pima_indians_unprocessed`)  
**Contract Version**: `v1` (Synchronized with `data-engineering/schema.yaml`)

---

## Predictor Features

### 1. `Pregnancies`
* **Type**: `int`
* **Range**: $[0, 17]$
* **Unit**: count
* **Definition**: Number of times pregnant.
* **Missing Value Rule**: Value $0$ is clinically valid (nulliparous women).

### 2. `Glucose`
* **Type**: `float`
* **Range**: $[44.0, 200.0]$
* **Unit**: $\text{mg/dL}$
* **Definition**: 2-hour plasma glucose concentration post oral glucose tolerance test (OGTT).
* **Missing Value Rule**: Value $0$ indicates a missing observation (converted to NaN and imputed).

### 3. `BloodPressure`
* **Type**: `float`
* **Range**: $[24.0, 122.0]$
* **Unit**: $\text{mm Hg}$
* **Definition**: Diastolic blood pressure.
* **Missing Value Rule**: Value $0$ indicates a missing observation (converted to NaN and imputed).

### 4. `SkinThickness`
* **Type**: `float`
* **Range**: $[7.0, 99.0]$
* **Unit**: $\text{mm}$
* **Definition**: Triceps skin fold thickness.
* **Missing Value Rule**: Value $0$ indicates a missing observation (converted to NaN and imputed).

### 5. `Insulin`
* **Type**: `float`
* **Range**: $[14.0, 846.0]$
* **Unit**: $\mu\text{U/ml}$
* **Definition**: 2-Hour serum insulin.
* **Missing Value Rule**: Value $0$ indicates a missing observation (converted to NaN and imputed).

### 6. `BMI`
* **Type**: `float`
* **Range**: $[18.0, 67.1]$
* **Unit**: $\text{kg/m}^2$
* **Definition**: Body mass index ($\text{weight in kg} / (\text{height in m})^2$).
* **Missing Value Rule**: Value $0$ indicates a missing observation (converted to NaN and imputed).

### 7. `DiabetesPedigreeFunction`
* **Type**: `float`
* **Range**: $[0.07, 2.42]$
* **Unit**: score
* **Definition**: Diabetes pedigree function (genetic score synthesizing diabetes family history).
* **Missing Value Rule**: Non-zero continuous metric.

### 8. `Age`
* **Type**: `int`
* **Range**: $[21, 81]$
* **Unit**: years
* **Definition**: Age in completed solar years at time of record capture.
* **Missing Value Rule**: Must be $\ge 21$.

---

## Target Feature

### `Outcome`
* **Type**: `binary`
* **Values**: $\{0, 1\}$
* **Positive Rate**: $34.9\%$ ($268 / 768$)
* **Definition**: $1$ when the patient is diagnosed with diabetes mellitus; $0$ otherwise.

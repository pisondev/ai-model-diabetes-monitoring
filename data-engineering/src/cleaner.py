"""
Data cleaning, missing value imputation, outlier handling, and feature engineering
for the Pima Indians Diabetes dataset.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
try:
    from .data_loader import ZERO_AS_MISSING_COLS
except (ImportError, ValueError):
    try:
        from src.data_loader import ZERO_AS_MISSING_COLS
    except ImportError:
        from data_loader import ZERO_AS_MISSING_COLS


def mark_hidden_missing_values(
    df: pd.DataFrame, cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Replace physiologically impossible 0 values with np.nan for clinical features.
    
    Args:
        df: Input DataFrame.
        cols: List of column names to check for hidden zero-missingness.
        
    Returns:
        pd.DataFrame: DataFrame with zeros replaced by NaN in specified columns.
    """
    df_clean = df.copy()
    target_cols = cols if cols is not None else ZERO_AS_MISSING_COLS
    
    for col in target_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)
            
    return df_clean


def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "median_by_outcome",
    n_neighbors: int = 5,
    impute_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Impute missing values using specified strategy.
    
    Strategies:
        - 'median_by_outcome': Impute median computed conditionally per target Outcome class.
        - 'knn': Multivariate k-Nearest Neighbors imputation.
        - 'median': Overall feature median (SimpleImputer).
        - 'mean': Overall feature mean (SimpleImputer).
        
    Args:
        df: DataFrame with NaNs.
        strategy: Imputation method.
        n_neighbors: Number of neighbors if strategy='knn'.
        impute_cols: Specific columns to impute (defaults to all columns with missing values).
        
    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    df_imputed = df.copy()
    if impute_cols is None:
        impute_cols = [col for col in df_imputed.columns if df_imputed[col].isna().any()]
    
    if not impute_cols:
        return df_imputed

    if strategy == "median_by_outcome" and "Outcome" in df_imputed.columns:
        for col in impute_cols:
            if col == "Outcome":
                continue
            # Calculate medians per outcome group
            medians = df_imputed.groupby("Outcome")[col].transform("median")
            df_imputed[col] = df_imputed[col].fillna(medians)
            # Fallback if any remain
            if df_imputed[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                
    elif strategy == "knn":
        # Separate non-numeric columns if any
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
        
    elif strategy in ("median", "mean"):
        imputer = SimpleImputer(strategy=strategy)
        df_imputed[impute_cols] = imputer.fit_transform(df_imputed[impute_cols])
        
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
        
    return df_imputed


def handle_outliers(
    df: pd.DataFrame,
    method: str = "winsorize",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    iqr_multiplier: float = 1.5,
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Handle outliers via quantile capping (winsorization) or IQR thresholding.
    
    Args:
        df: Input DataFrame.
        method: 'winsorize' (quantile-based capping) or 'iqr_cap' or 'none'.
        lower_quantile: Lower quantile cutoff for winsorizing (e.g., 0.01).
        upper_quantile: Upper quantile cutoff for winsorizing (e.g., 0.99).
        iqr_multiplier: Multiplier for IQR rule.
        cols: Columns to apply outlier treatment to (defaults to continuous features).
        
    Returns:
        pd.DataFrame: Outlier-treated DataFrame.
    """
    if method == "none":
        return df.copy()
        
    df_treated = df.copy()
    if cols is None:
        cols = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
    
    for col in cols:
        if col not in df_treated.columns:
            continue
            
        series = df_treated[col].dropna()
        if series.empty:
            continue
            
        if method == "winsorize":
            lower_bound = series.quantile(lower_quantile)
            upper_bound = series.quantile(upper_quantile)
        elif method == "iqr_cap":
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = max(0.0, q25 - iqr_multiplier * iqr)
            upper_bound = q75 + iqr_multiplier * iqr
        else:
            raise ValueError(f"Unknown outlier treatment method: {method}")
            
        df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
        
    return df_treated


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive clinically motivated domain features for diabetes risk modeling.
    
    Added Features:
        - BMI_Category: WHO standard categories (Underweight, Normal, Overweight, Obese)
        - Glucose_Risk: Normal (<140), Prediabetes (140-199), Diabetes_Range (>=200)
        - Age_Group: Young (<30), Middle (30-49), Senior (>=50)
        - Pregnancy_Rate: Pregnancies / Age
        - HOMA_IR_Proxy: Glucose * Insulin / 405.0 (insulin resistance index proxy)
        - Metabolic_Risk_Score: Integer count (0-4) of elevated metabolic risk markers
        
    Args:
        df: Cleaned and imputed DataFrame.
        
    Returns:
        pd.DataFrame: Enriched DataFrame.
    """
    df_eng = df.copy()
    
    # 1. WHO BMI Categorization
    if "BMI" in df_eng.columns:
        df_eng["BMI_Category"] = pd.cut(
            df_eng["BMI"],
            bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
        )
        
    # 2. Oral Glucose Tolerance Test (OGTT) Risk Tiers
    if "Glucose" in df_eng.columns:
        df_eng["Glucose_Risk"] = pd.cut(
            df_eng["Glucose"],
            bins=[-np.inf, 139.9, 199.9, np.inf],
            labels=["Normal", "Prediabetes", "Diabetes_Range"],
        )
        
    # 3. Age Grouping
    if "Age" in df_eng.columns:
        df_eng["Age_Group"] = pd.cut(
            df_eng["Age"],
            bins=[-np.inf, 29.9, 49.9, np.inf],
            labels=["Young", "Middle", "Senior"],
        )
        
    # 4. Pregnancy Rate (Fertility-Age interaction)
    if "Pregnancies" in df_eng.columns and "Age" in df_eng.columns:
        df_eng["Pregnancy_Rate"] = df_eng["Pregnancies"] / df_eng["Age"]
        
    # 5. HOMA-IR Proxy (Homeostatic Model Assessment for Insulin Resistance)
    # Standard formula: (Fasting Glucose (mg/dL) * Fasting Insulin (uU/mL)) / 405
    if "Glucose" in df_eng.columns and "Insulin" in df_eng.columns:
        df_eng["HOMA_IR_Proxy"] = (df_eng["Glucose"] * df_eng["Insulin"]) / 405.0
        
    # 6. Metabolic Risk Score
    risk_score = pd.Series(0, index=df_eng.index)
    if "Glucose" in df_eng.columns:
        risk_score += (df_eng["Glucose"] >= 140).astype(int)
    if "BloodPressure" in df_eng.columns:
        risk_score += (df_eng["BloodPressure"] >= 85).astype(int)
    if "BMI" in df_eng.columns:
        risk_score += (df_eng["BMI"] >= 30.0).astype(int)
    if "Age" in df_eng.columns:
        risk_score += (df_eng["Age"] >= 45).astype(int)
    df_eng["Metabolic_Risk_Score"] = risk_score
    
    return df_eng


def scale_features(
    df: pd.DataFrame,
    scaler_type: str = "robust",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Scale numeric features using StandardScaler, RobustScaler, or MinMaxScaler.
    
    Args:
        df: Input DataFrame.
        scaler_type: 'standard', 'robust', or 'minmax'.
        exclude_cols: Columns to leave unscaled (e.g. 'Outcome', categorical features).
        
    Returns:
        Tuple[pd.DataFrame, scaler_object]
    """
    df_scaled = df.copy()
    if exclude_cols is None:
        exclude_cols = ["Outcome"]
        
    numeric_cols = [
        c for c in df_scaled.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
        
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled, scaler

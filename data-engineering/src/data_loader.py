"""
Data loader and schema validation utilities for Pima Indians Diabetes dataset.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Features where 0 is physiologically invalid and indicates a missing observation
ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "Pregnancies": {
        "description": "Number of times pregnant",
        "unit": "count",
        "zero_valid": True,
        "dtype": "int64",
    },
    "Glucose": {
        "description": "Plasma glucose concentration 2h after oral glucose tolerance test",
        "unit": "mg/dL",
        "zero_valid": False,
        "dtype": "float64",
    },
    "BloodPressure": {
        "description": "Diastolic blood pressure",
        "unit": "mm Hg",
        "zero_valid": False,
        "dtype": "float64",
    },
    "SkinThickness": {
        "description": "Triceps skin fold thickness",
        "unit": "mm",
        "zero_valid": False,
        "dtype": "float64",
    },
    "Insulin": {
        "description": "2-Hour serum insulin",
        "unit": "mu U/ml",
        "zero_valid": False,
        "dtype": "float64",
    },
    "BMI": {
        "description": "Body mass index (weight in kg/(height in m)^2)",
        "unit": "kg/m^2",
        "zero_valid": False,
        "dtype": "float64",
    },
    "DiabetesPedigreeFunction": {
        "description": "Diabetes pedigree function (family history risk score)",
        "unit": "score",
        "zero_valid": True,
        "dtype": "float64",
    },
    "Age": {
        "description": "Age of the patient",
        "unit": "years",
        "zero_valid": False,
        "dtype": "int64",
    },
    "Outcome": {
        "description": "Class variable (0: Non-diabetic, 1: Diabetic)",
        "unit": "binary",
        "zero_valid": True,
        "dtype": "int64",
    },
}


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw diabetes dataset from CSV.
    
    Args:
        filepath: Path to raw CSV file. Defaults to looking in data-engineering/raw or raw.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if filepath is None:
        candidates = [
            Path("data-engineering/raw/diabetes.csv"),
            Path("raw/diabetes.csv"),
            Path("../raw/diabetes.csv"),
            Path("data/raw/diabetes.csv"),
        ]
        for c in candidates:
            if c.exists():
                filepath = str(c)
                break
        if filepath is None:
            filepath = "data-engineering/raw/diabetes.csv"
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")
    
    df = pd.read_csv(path)
    validate_raw_schema(df)
    return df


def validate_raw_schema(df: pd.DataFrame) -> bool:
    """
    Validate columns, row count, and types against expected Pima dataset schema.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        bool: True if schema is valid.
    """
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing expected columns: {missing_cols}")
    
    if len(df) == 0:
        raise ValueError("Dataset is empty.")
    
    if not set(df["Outcome"].unique()).issubset({0, 1}):
        raise ValueError("Target 'Outcome' must contain only 0 and 1 values.")
        
    return True

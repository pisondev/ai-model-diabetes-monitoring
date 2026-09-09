"""
End-to-end orchestration pipeline for Pima Indians Diabetes EDA and Data Cleaning.
"""

import argparse
from pathlib import Path
from tabulate import tabulate
import pandas as pd

try:
    from .data_loader import load_raw_data, validate_raw_schema
    from .cleaner import (
        mark_hidden_missing_values,
        impute_missing_values,
        handle_outliers,
        engineer_features,
        scale_features,
    )
    from .eda_analyzer import (
        generate_summary_statistics,
        compute_missingness_report,
        compute_bivariate_stats,
        detect_outliers_iqr,
        save_eda_figures,
    )
except (ImportError, ValueError):
    try:
        from src.data_loader import load_raw_data, validate_raw_schema
        from src.cleaner import (
            mark_hidden_missing_values,
            impute_missing_values,
            handle_outliers,
            engineer_features,
            scale_features,
        )
        from src.eda_analyzer import (
            generate_summary_statistics,
            compute_missingness_report,
            compute_bivariate_stats,
            detect_outliers_iqr,
            save_eda_figures,
        )
    except ImportError:
        from data_loader import load_raw_data, validate_raw_schema
        from cleaner import (
            mark_hidden_missing_values,
            impute_missing_values,
            handle_outliers,
            engineer_features,
            scale_features,
        )
        from eda_analyzer import (
            generate_summary_statistics,
            compute_missingness_report,
            compute_bivariate_stats,
            detect_outliers_iqr,
            save_eda_figures,
        )


def _resolve_default_paths():
    base = Path("data-engineering") if Path("data-engineering").is_dir() else Path(".")
    return {
        "raw": str(base / "raw" / "diabetes.csv"),
        "cleaned": str(base / "processed" / "dataset_m2_v1.csv"),
        "engineered": str(base / "processed" / "dataset_m2_v1_engineered.csv"),
        "figures": str(base / "reports" / "figures"),
    }


def run_pipeline(
    raw_path: str = None,
    output_cleaned_path: str = None,
    output_engineered_path: str = None,
    imputation_strategy: str = "median_by_outcome",
    outlier_method: str = "winsorize",
    save_figures: bool = True,
    figures_dir: str = None,
) -> None:
    """
    Execute end-to-end EDA and cleaning pipeline.
    """
    defaults = _resolve_default_paths()
    raw_path = raw_path or defaults["raw"]
    output_cleaned_path = output_cleaned_path or defaults["cleaned"]
    output_engineered_path = output_engineered_path or defaults["engineered"]
    figures_dir = figures_dir or defaults["figures"]

    print("=" * 80)
    print(" 🩺 PIMA INDIANS DIABETES: EDA & DATA CLEANING PIPELINE")
    print("=" * 80)
    
    # 1. Ingestion & Schema Validation
    print(f"\n[1/6] Loading raw dataset from: {raw_path}")
    df_raw = load_raw_data(raw_path)
    print(f"      Loaded {df_raw.shape[0]} rows and {df_raw.shape[1]} columns successfully.")
    
    # 2. Raw Profile & Missingness Diagnostics
    print("\n[2/6] Profiling raw data & identifying hidden zero-missing values...")
    missing_report = compute_missingness_report(df_raw)
    print("\n--- Hidden Missing Value Report ---")
    print(tabulate(missing_report, headers="keys", tablefmt="fancy_grid"))
    
    # Convert hidden zeros to NaN
    df_nan = mark_hidden_missing_values(df_raw)
    nan_summary = generate_summary_statistics(df_nan)
    print("\n--- Summary Statistics (Post Zero-to-NaN Conversion) ---")
    print(tabulate(nan_summary, headers="keys", tablefmt="fancy_grid"))
    
    # Bivariate analysis
    bivariate_df = compute_bivariate_stats(df_nan)
    print("\n--- Bivariate Analysis & Mann-Whitney U Hypothesis Tests ---")
    print(tabulate(bivariate_df, headers="keys", tablefmt="fancy_grid"))
    
    # Outliers
    outlier_df = detect_outliers_iqr(df_nan)
    print("\n--- Outlier Detection (1.5 x IQR Fences) ---")
    print(tabulate(outlier_df, headers="keys", tablefmt="fancy_grid"))
    
    # 3. Data Cleaning & Imputation
    print(f"\n[3/6] Applying imputation strategy: '{imputation_strategy}'...")
    df_imputed = impute_missing_values(df_nan, strategy=imputation_strategy)
    
    print(f"      Applying outlier treatment: '{outlier_method}' (1st - 99th percentile capping)...")
    df_cleaned = handle_outliers(df_imputed, method=outlier_method)
    
    # Validate cleaning
    assert not df_cleaned.isna().any().any(), "Error: Cleaned dataset contains lingering NaNs!"
    
    # 4. Feature Engineering
    print("\n[4/6] Engineering domain-specific features (WHO BMI, Glucose Risk, HOMA-IR Proxy, Pregnancy Rate)...")
    df_engineered = engineer_features(df_cleaned)
    print(f"      Engineered dataset expanded to {df_engineered.shape[1]} features.")
    
    # 5. Saving Artifacts
    print("\n[5/6] Exporting processed datasets...")
    Path(output_cleaned_path).parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_cleaned_path, index=False)
    print(f"      Cleaned dataset saved to: {output_cleaned_path} (Shape: {df_cleaned.shape})")
    
    Path(output_engineered_path).parent.mkdir(parents=True, exist_ok=True)
    df_engineered.to_csv(output_engineered_path, index=False)
    print(f"      Engineered dataset saved to: {output_engineered_path} (Shape: {df_engineered.shape})")
    
    # 6. Generating Visualizations
    if save_figures:
        print(f"\n[6/6] Generating visualization figures in '{figures_dir}'...")
        saved_figs = save_eda_figures(
            df_raw=df_raw,
            df_nan=df_nan,
            df_cleaned=df_cleaned,
            df_engineered=df_engineered,
            output_dir=figures_dir,
        )
        for fig in saved_figs:
            print(f"      Generated: {fig}")
            
    print("\n" + "=" * 80)
    print(" ✅ EDA AND DATA CLEANING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pima Diabetes EDA and Cleaning Pipeline")
    parser.add_argument("--raw-path", default=None, help="Path to raw CSV")
    parser.add_argument("--output-cleaned", default=None, help="Path to save cleaned CSV")
    parser.add_argument("--output-engineered", default=None, help="Path to save engineered CSV")
    parser.add_argument("--impute-strategy", default="median_by_outcome", choices=["median_by_outcome", "knn", "median", "mean"], help="Imputation strategy")
    parser.add_argument("--outlier-method", default="winsorize", choices=["winsorize", "iqr_cap", "none"], help="Outlier handling method")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--figures-dir", default=None, help="Directory to save figures")
    
    args = parser.parse_args()
    
    run_pipeline(
        raw_path=args.raw_path,
        output_cleaned_path=args.output_cleaned,
        output_engineered_path=args.output_engineered,
        imputation_strategy=args.impute_strategy,
        outlier_method=args.outlier_method,
        save_figures=not args.no_figures,
        figures_dir=args.figures_dir,
    )

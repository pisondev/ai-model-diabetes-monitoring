"""
Exploratory Data Analysis (EDA) engine for the Pima Indians Diabetes dataset.
Produces statistical summaries, hypothesis testing, missingness diagnostics, and figures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

try:
    from .data_loader import ZERO_AS_MISSING_COLS
except (ImportError, ValueError):
    try:
        from src.data_loader import ZERO_AS_MISSING_COLS
    except ImportError:
        from data_loader import ZERO_AS_MISSING_COLS


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive numerical summary including skewness, kurtosis, and zero counts.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: Statistical profile table.
    """
    stats_list = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        series = df[col]
        non_null = series.dropna()
        
        zero_cnt = int((series == 0).sum())
        zero_pct = float(zero_cnt / len(series) * 100)
        nan_cnt = int(series.isna().sum())
        nan_pct = float(nan_cnt / len(series) * 100)
        
        skew_val = float(stats.skew(non_null)) if len(non_null) > 2 else np.nan
        kurt_val = float(stats.kurtosis(non_null)) if len(non_null) > 2 else np.nan
        
        stats_list.append({
            "Feature": col,
            "Count": int(series.count()),
            "Mean": round(float(series.mean()), 3),
            "Std": round(float(series.std()), 3),
            "Min": round(float(series.min()), 3),
            "25%": round(float(series.quantile(0.25)), 3),
            "Median (50%)": round(float(series.median()), 3),
            "75%": round(float(series.quantile(0.75)), 3),
            "Max": round(float(series.max()), 3),
            "Skewness": round(skew_val, 3),
            "Kurtosis": round(kurt_val, 3),
            "Zero_Count": zero_cnt,
            "Zero_Pct(%)": round(zero_pct, 2),
            "NaN_Count": nan_cnt,
            "NaN_Pct(%)": round(nan_pct, 2),
        })
        
    return pd.DataFrame(stats_list).set_index("Feature")


def compute_missingness_report(
    raw_df: pd.DataFrame, zero_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze hidden zero-missingness vs explicit missingness.
    
    Args:
        raw_df: Raw DataFrame before zero-replacement.
        zero_cols: Columns where 0 indicates missingness.
        
    Returns:
        pd.DataFrame: Missingness breakdown.
    """
    if zero_cols is None:
        zero_cols = ZERO_AS_MISSING_COLS
        
    rows = []
    for col in raw_df.columns:
        if col == "Outcome":
            continue
        zero_count = int((raw_df[col] == 0).sum())
        is_hidden_missing = col in zero_cols
        missing_count = zero_count if is_hidden_missing else int(raw_df[col].isna().sum())
        missing_pct = float(missing_count / len(raw_df) * 100)
        
        rows.append({
            "Feature": col,
            "Total_Rows": len(raw_df),
            "Zero_Count": zero_count,
            "Is_Physiologically_Missing": is_hidden_missing,
            "True_Missing_Count": missing_count,
            "Missing_Percentage": round(missing_pct, 2),
        })
        
    return pd.DataFrame(rows).set_index("Feature")


def compute_bivariate_stats(
    df: pd.DataFrame, target_col: str = "Outcome"
) -> pd.DataFrame:
    """
    Compute stratified statistics and Mann-Whitney U test p-values across target classes.
    
    Args:
        df: Input DataFrame with target column.
        target_col: Target binary column.
        
    Returns:
        pd.DataFrame: Bivariate comparison with statistical significance tests.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    df_0 = df[df[target_col] == 0]
    df_1 = df[df[target_col] == 1]
    
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    
    results = []
    for col in feature_cols:
        s0 = df_0[col].dropna()
        s1 = df_1[col].dropna()
        
        # Mann-Whitney U test (non-parametric two-sample test)
        stat, p_val = stats.mannwhitneyu(s0, s1, alternative="two-sided")
        
        results.append({
            "Feature": col,
            "NonDiabetic_Mean": round(float(s0.mean()), 2),
            "NonDiabetic_Median": round(float(s0.median()), 2),
            "NonDiabetic_IQR": round(float(s0.quantile(0.75) - s0.quantile(0.25)), 2),
            "Diabetic_Mean": round(float(s1.mean()), 2),
            "Diabetic_Median": round(float(s1.median()), 2),
            "Diabetic_IQR": round(float(s1.quantile(0.75) - s1.quantile(0.25)), 2),
            "MannWhitney_U_Stat": round(float(stat), 1),
            "p_value": f"{p_val:.2e}" if p_val < 0.001 else round(float(p_val), 4),
            "Statistically_Significant (p<0.05)": p_val < 0.05,
        })
        
    return pd.DataFrame(results).set_index("Feature")


def detect_outliers_iqr(
    df: pd.DataFrame, multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Compute IQR-based lower and upper fences and count outliers per numeric feature.
    
    Args:
        df: Input DataFrame.
        multiplier: Multiplier for IQR rule.
        
    Returns:
        pd.DataFrame: Outlier boundary and count table.
    """
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Outcome"]
    outlier_list = []
    
    for col in numeric_cols:
        series = df[col].dropna()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_fence = q25 - multiplier * iqr
        upper_fence = q75 + multiplier * iqr
        
        outliers = series[(series < lower_fence) | (series > upper_fence)]
        
        outlier_list.append({
            "Feature": col,
            "Q25": round(float(q25), 2),
            "Q75": round(float(q75), 2),
            "IQR": round(float(iqr), 2),
            "Lower_Fence": round(float(lower_fence), 2),
            "Upper_Fence": round(float(upper_fence), 2),
            "Outlier_Count": len(outliers),
            "Outlier_Pct(%)": round(len(outliers) / len(series) * 100, 2),
            "Min_Value": round(float(series.min()), 2),
            "Max_Value": round(float(series.max()), 2),
        })
        
    return pd.DataFrame(outlier_list).set_index("Feature")


def save_eda_figures(
    df_raw: pd.DataFrame,
    df_nan: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    df_engineered: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> List[str]:
    """
    Generate and save publication-quality visual EDA charts.
    
    Args:
        df_raw: Raw dataset.
        df_nan: Dataset with zeros converted to NaN.
        df_cleaned: Cleaned and imputed dataset.
        df_engineered: Dataset with engineered features.
        output_dir: Directory to save figure images.
        
    Returns:
        List[str]: Paths to generated image files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    
    # 1. Missingness Profile (Hidden vs Valid Zeros)
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_pcts = (df_nan[ZERO_AS_MISSING_COLS].isna().mean() * 100).sort_values(ascending=False)
    colors = ["#d9534f" if p > 20 else "#f0ad4e" if p > 5 else "#5cb85c" for p in missing_pcts]
    
    bars = ax.barh(missing_pcts.index, missing_pcts.values, color=colors, edgecolor="black", height=0.6)
    ax.set_xlabel("Missing Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Hidden Missing Data (Physiologically Invalid Zeros)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 60)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.0, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontweight="bold")
    plt.tight_layout()
    f1 = out_path / "01_missingness_profile.png"
    plt.savefig(f1, dpi=200)
    plt.close()
    generated_files.append(str(f1))
    
    # 2. Target Class Balance
    fig, ax = plt.subplots(figsize=(6, 5))
    outcome_counts = df_raw["Outcome"].value_counts()
    labels = ["Non-Diabetic (0)", "Diabetic (1)"]
    palette_pie = ["#5bc0de", "#d9534f"]
    wedges, texts, autotexts = ax.pie(
        outcome_counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=palette_pie,
        explode=[0, 0.05],
        wedgeprops=dict(edgecolor="black", linewidth=1.2),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Outcome Class Distribution (Class Balance)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    f2 = out_path / "02_class_balance.png"
    plt.savefig(f2, dpi=200)
    plt.close()
    generated_files.append(str(f2))
    
    # 3. Univariate Distributions Before vs After Cleaning (KDE & Histograms)
    features_to_plot = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(features_to_plot):
        ax = axes[i]
        sns.kdeplot(
            data=df_raw[col],
            ax=ax,
            label="Raw (with 0s)",
            color="#d9534f",
            linestyle="--",
            fill=False,
            linewidth=1.8,
        )
        sns.kdeplot(
            data=df_cleaned[col],
            ax=ax,
            label="Cleaned & Imputed",
            color="#0275d8",
            fill=True,
            alpha=0.3,
            linewidth=2.0,
        )
        ax.set_title(f"Distribution: {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col)
        ax.legend()
        
    plt.suptitle("Feature Distributions: Raw vs Cleaned/Imputed", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    f3 = out_path / "03_distributions_raw_vs_cleaned.png"
    plt.savefig(f3, dpi=200)
    plt.close()
    generated_files.append(str(f3))
    
    # 4. Boxplots Stratified by Diabetes Outcome
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
    axes = axes.flatten()
    numeric_features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    
    for i, col in enumerate(numeric_features):
        ax = axes[i]
        sns.boxplot(
            x="Outcome",
            y=col,
            hue="Outcome",
            data=df_cleaned,
            ax=ax,
            palette=["#5bc0de", "#d9534f"],
            legend=False,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "yellow", "markeredgecolor": "black"},
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-Diabetic (0)", "Diabetic (1)"])
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        
    plt.suptitle("Clinical Feature Separation by Diabetes Outcome (Yellow Dot = Mean)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    f4 = out_path / "04_boxplots_by_outcome.png"
    plt.savefig(f4, dpi=200)
    plt.close()
    generated_files.append(str(f4))
    
    # 5. Correlation Heatmap (Pearson & Spearman)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    corr_pearson = df_cleaned[numeric_features + ["Outcome"]].corr(method="pearson")
    corr_spearman = df_cleaned[numeric_features + ["Outcome"]].corr(method="spearman")
    
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    
    sns.heatmap(
        corr_pearson,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.4,
        vmax=0.8,
        mask=mask,
        ax=ax1,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    ax1.set_title("Pearson Correlation Matrix (Linear)", fontsize=13, fontweight="bold")
    
    sns.heatmap(
        corr_spearman,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-0.4,
        vmax=0.8,
        mask=mask,
        ax=ax2,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    ax2.set_title("Spearman Rank Correlation Matrix (Monotonic)", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    f5 = out_path / "05_correlation_heatmaps.png"
    plt.savefig(f5, dpi=200)
    plt.close()
    generated_files.append(str(f5))
    
    # 6. Engineered Features Impact Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Glucose Risk vs Outcome
    if "Glucose_Risk" in df_engineered.columns:
        cross_glucose = pd.crosstab(df_engineered["Glucose_Risk"], df_engineered["Outcome"], normalize="index") * 100
        cross_glucose.plot(kind="bar", stacked=True, color=["#5bc0de", "#d9534f"], ax=axes[0], edgecolor="black")
        axes[0].set_title("Glucose Risk Category vs Outcome (%)", fontweight="bold")
        axes[0].set_ylabel("Percentage (%)")
        axes[0].set_xlabel("")
        axes[0].legend(["Non-Diabetic", "Diabetic"], loc="upper left")
        axes[0].tick_params(axis="x", rotation=15)
        
    # BMI Category vs Outcome
    if "BMI_Category" in df_engineered.columns:
        cross_bmi = pd.crosstab(df_engineered["BMI_Category"], df_engineered["Outcome"], normalize="index") * 100
        cross_bmi.plot(kind="bar", stacked=True, color=["#5bc0de", "#d9534f"], ax=axes[1], edgecolor="black")
        axes[1].set_title("WHO BMI Category vs Outcome (%)", fontweight="bold")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_xlabel("")
        axes[1].legend(["Non-Diabetic", "Diabetic"], loc="upper left")
        axes[1].tick_params(axis="x", rotation=15)
        
    # HOMA-IR Proxy KDE
    if "HOMA_IR_Proxy" in df_engineered.columns:
        sns.kdeplot(
            data=df_engineered[df_engineered["Outcome"] == 0]["HOMA_IR_Proxy"],
            ax=axes[2],
            label="Non-Diabetic (0)",
            color="#5bc0de",
            fill=True,
            alpha=0.4,
        )
        sns.kdeplot(
            data=df_engineered[df_engineered["Outcome"] == 1]["HOMA_IR_Proxy"],
            ax=axes[2],
            label="Diabetic (1)",
            color="#d9534f",
            fill=True,
            alpha=0.4,
        )
        axes[2].set_title("HOMA-IR Insulin Resistance Proxy Distribution", fontweight="bold")
        axes[2].set_xlabel("HOMA-IR Proxy Value")
        axes[2].legend()
        
    plt.suptitle("Clinical Feature Engineering Diagnostics", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    f6 = out_path / "06_engineered_features_diagnostics.png"
    plt.savefig(f6, dpi=200)
    plt.close()
    generated_files.append(str(f6))
    
    return generated_files

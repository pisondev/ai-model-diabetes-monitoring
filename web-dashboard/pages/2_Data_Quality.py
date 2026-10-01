import streamlit as st

from lib.charts import completeness_bar, correlation_heatmap, numeric_boxplots, positive_rate_by_level
from lib.config import numeric_features, target_name
from lib.data import derived_column, implausible_zeros, imputation_spikes, load_dataset
from lib.ui import figure, page, source_badge

page("Cohort data quality", "Evidence behind the cleaning decisions the data side made")

frame, source = load_dataset()
source_badge(source)

target = target_name()
numeric = numeric_features()
zeros = implausible_zeros(frame)

duplicates, missing, impossible = st.columns(3)
duplicates.metric("Duplicate patients", f"{int(frame.duplicated().sum()):,}")
missing.metric("Cells missing", f"{int(frame.isna().sum().sum()):,}")
impossible.metric("Impossible zeros left", f"{sum(zeros.values()):,}")

st.caption(
    "A zero is impossible in "
    + ", ".join(zeros)
    + " because the contract puts each of their minima above zero. In the raw download those "
    "zeros were missing readings in disguise; the cleaning log records how each was resolved."
)

figure(
    "Completeness per field",
    completeness_bar(frame),
    "Declared nulls only. The count above is the check that the disguised kind was resolved too.",
)

st.subheader("Values that carry the diagnosis")
spikes = imputation_spikes(frame, target)
if spikes:
    st.warning(
        "The cleaning step filled each missing reading with the median of that patient's own "
        "Outcome class. Every refilled cell therefore carries the answer, and a model trained on "
        "this frame can read the label straight out of the feature."
    )
    st.dataframe(
        [
            {
                "Field": row["field"],
                "Value": row["value"],
                "Patients": row["patients"],
                "Share of cohort": f"{row['share']:.1%}",
                "Diabetic": f"{row['positive_rate']:.1%}",
            }
            for row in spikes
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Detected live: an exact value held by at least 5% of the cohort whose diabetic share "
        f"sits at least 25 points from the cohort's {frame[target].mean():.1%}. The Insulin and "
        "SkinThickness rows are the fill values. A row such as the youngest age in the cohort is a "
        "real pattern rather than an artefact, so the table is read rather than trusted blindly."
    )
    st.info(
        "Consequence for modeling, agreed with the modeling side: training reads the raw file with "
        "zeros as missing, and imputation happens inside the model pipeline, fitted on the training "
        "folds without the label. This frozen frame stays the profiling and reporting artefact. "
        "Measured on this cohort, the shortcut inflates Random Forest PR-AUC from 0.722 to 0.913 "
        "and recall from 0.634 to 0.817."
    )
else:
    st.success("No single value concentrates the diagnosis, so no imputation artefact is visible.")

figure(
    "Spread and outliers per clinical field",
    numeric_boxplots(frame, numeric),
    "Box is the interquartile range, whisker reaches 1.5 IQR, tooltip carries the count beyond it. "
    "Each panel keeps its own axis because the fields do not share a scale.",
    container_width=False,
)

figure(
    "Correlation between clinical fields and the diagnosis",
    correlation_heatmap(frame, numeric + [target]),
    "Read the diagnosis row first. No single reading restates the label here, which is the "
    "difference between this cohort and a dataset where the label is definitional.",
)

body_band = derived_column(frame, "body_band")
glucose_band = derived_column(frame, "glucose_band")
if body_band and glucose_band:
    body, glucose = st.columns(2)
    with body:
        figure(
            "Diabetic share by body-mass band",
            positive_rate_by_level(frame, body_band, target, order=["Normal", "Overweight", "Obese"]),
            "WHO bands, assigned by the cleaning pipeline rather than by this application.",
        )
    with glucose:
        figure(
            "Diabetic share by glucose band",
            positive_rate_by_level(frame, glucose_band, target, order=["Normal", "Prediabetes"]),
            "OGTT thresholds. The gap between the two bands is the strongest single split in the cohort.",
        )
else:
    st.info(
        "The derived clinical bands are not in the loaded frame. Load the engineered cohort to see "
        "the diabetic share per band."
    )

st.info(
    "Before and after cleaning evidence is produced in 02_data_engineering_lab.ipynb and written up "
    "in data-engineering/reports/. This page reads whichever dataset version is present and says "
    "which one at the top."
)

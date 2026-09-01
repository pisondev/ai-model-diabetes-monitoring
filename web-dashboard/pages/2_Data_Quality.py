import streamlit as st

from lib.charts import completeness_bar, correlation_heatmap, numeric_boxplots, positive_rate_by_category
from lib.config import categorical_features, numeric_features, target_name
from lib.data import load_dataset
from lib.ui import figure, page, source_badge

page("Data quality", "Evidence that supports the cleaning decisions")

frame, source = load_dataset()
source_badge(source)

target = target_name()
numeric = numeric_features()

duplicates, missing, sentinel = st.columns(3)
duplicates.metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
missing.metric("Cells missing", f"{int(frame.isna().sum().sum()):,}")
sentinel.metric("Recorded as No Info", f"{int((frame['smoking_history'] == 'No Info').sum()):,}")

figure(
    "Completeness per field",
    completeness_bar(frame),
    "Declared nulls only. The No Info count above is missingness the schema records as a value, so "
    "it never appears in this chart and has to be treated separately.",
)

figure(
    "Spread and outliers per numeric field",
    numeric_boxplots(frame, numeric),
    "Box is the interquartile range, whisker reaches 1.5 IQR, and the tooltip carries the count "
    "beyond it. Each panel keeps its own axis because the fields do not share a scale.",
    container_width=False,
)

figure(
    "Correlation between numeric fields and the label",
    correlation_heatmap(frame, numeric + [target]),
    "Read the label row first. A coefficient close to the diagonal is the warning that a feature "
    "restates the label rather than predicting it.",
)

gender, smoking = st.columns(2)
with gender:
    figure(
        "Positive rate by gender",
        positive_rate_by_category(frame, "gender", target),
        "The grey rule is the rate across the whole dataset.",
    )
with smoking:
    figure(
        "Positive rate by smoking history",
        positive_rate_by_category(frame, "smoking_history", target),
        "Where No Info sits against the rule says whether the missingness is informative.",
    )

st.caption(f"Categorical fields in the contract: {', '.join(categorical_features())}")

st.info(
    "Before and after cleaning evidence is produced in 02_data_engineering_lab.ipynb. "
    "This page reads whatever dataset version is present and stays honest about which one."
)

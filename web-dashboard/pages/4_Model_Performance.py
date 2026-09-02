import streamlit as st

from lib.charts import accuracy_vs_prevalence_line, majority_baseline_bar, positive_rate_by_level
from lib.config import target_name
from lib.data import derived_column, load_dataset
from lib.ui import figure, page, source_badge

page("Model performance", "The floor a trained model has to clear")

frame, source = load_dataset()
source_badge(source)

target = target_name()
prevalence = frame[target].mean()
score_column = derived_column(frame, "risk_score")

st.info(
    "No experiment has been published yet. Until one is, this page scores the rules that already "
    "exist against the loaded cohort. Every number here is computed, none is assumed."
)

floor, warning = st.columns(2)
with floor:
    figure(
        "The always-negative rule scored on this cohort",
        majority_baseline_bar(prevalence),
        "A model that never predicts diabetes. It finds nobody, and it still reports the accuracy "
        "on the left. That number is the one to beat and the one to distrust.",
    )
with warning:
    figure(
        "Why accuracy is the wrong headline",
        accuracy_vs_prevalence_line(prevalence),
        "Accuracy of that same rule as prevalence moves. The marked point is this cohort, which "
        "is why the table below leads with PR-AUC and recall.",
    )

if score_column:
    figure(
        "The rule-based score already separates the cohort",
        positive_rate_by_level(frame, score_column, target),
        "Diabetic share at each level of the derived metabolic risk score. A trained model has to "
        "beat this ordering, not just the always-negative rule, before it is worth deploying.",
    )

st.subheader("Metrics to publish")
st.table(
    [
        {"Metric": "PR-AUC", "Majority baseline": f"{prevalence:.3f}", "Trained model": "pending"},
        {"Metric": "Recall", "Majority baseline": "0.000", "Trained model": "pending"},
        {"Metric": "F1", "Majority baseline": "0.000", "Trained model": "pending"},
        {"Metric": "ROC-AUC", "Majority baseline": "0.500", "Trained model": "pending"},
        {"Metric": "Accuracy", "Majority baseline": f"{1 - prevalence:.3f}", "Trained model": "pending"},
    ]
)

st.caption(
    "The baseline column is computed from the cohort currently loaded. When the modeling side "
    "publishes a metrics file, the trained column reads from it and the placeholder disappears."
)

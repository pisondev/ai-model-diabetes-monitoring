import streamlit as st

from lib.charts import accuracy_vs_prevalence_line, majority_baseline_bar
from lib.config import target_name
from lib.data import load_dataset
from lib.ui import figure, page, source_badge

page("Model performance", "Baseline against hybrid ensemble")

frame, source = load_dataset()
source_badge(source)

positive_rate = frame[target_name()].mean()

st.info(
    "No experiment has been published yet. Until one is, this page shows the floor the trained "
    "model has to clear, computed from the dataset rather than assumed."
)

floor, warning = st.columns(2)
with floor:
    figure(
        "The always-negative rule scored on this dataset",
        majority_baseline_bar(positive_rate),
        "A model that never predicts the positive class. It finds nobody, and it still reports the "
        "accuracy on the left. That number is the one to beat and the one to distrust.",
    )
with warning:
    figure(
        "Why accuracy is the wrong headline here",
        accuracy_vs_prevalence_line(positive_rate),
        "Accuracy of that same rule as the positive rate moves. The marked point is this dataset, "
        "so the metric table below leads with PR-AUC and recall instead.",
    )

st.subheader("Metrics to publish")
st.table(
    [
        {"Metric": "PR-AUC", "Majority baseline": f"{positive_rate:.3f}", "Trained model": "pending"},
        {"Metric": "Recall", "Majority baseline": "0.000", "Trained model": "pending"},
        {"Metric": "F1", "Majority baseline": "0.000", "Trained model": "pending"},
        {"Metric": "ROC-AUC", "Majority baseline": "0.500", "Trained model": "pending"},
        {"Metric": "Accuracy", "Majority baseline": f"{1 - positive_rate:.3f}", "Trained model": "pending"},
    ]
)

st.caption(
    "Baseline column is computed from the dataset currently loaded. When the modeling side "
    "publishes a metrics file, the trained column reads from it and the placeholder disappears."
)

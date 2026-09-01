import streamlit as st

from lib.ui import page

page("Model performance", "Baseline against hybrid ensemble")

st.info(
    "No experiment has been published yet. When the modeling side lands, this page reads its "
    "metrics file and the placeholder disappears."
)

st.table(
    [
        {"Metric": "PR-AUC", "Baseline": "pending", "Hybrid": "pending"},
        {"Metric": "Recall", "Baseline": "pending", "Hybrid": "pending"},
        {"Metric": "F1", "Baseline": "pending", "Hybrid": "pending"},
        {"Metric": "ROC-AUC", "Baseline": "pending", "Hybrid": "pending"},
    ]
)

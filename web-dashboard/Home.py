import streamlit as st

from lib.charts import category_bar, category_donut, class_balance_bar
from lib.config import APP_SUBTITLE, APP_TITLE, dataset_config, target_name
from lib.data import load_dataset
from lib.ui import disclaimer, figure, page, source_badge

page(APP_TITLE, APP_SUBTITLE)

frame, source = load_dataset()
source_badge(source)

target = target_name()
positive_rate = frame[target].mean()

records, features, positives, complete = st.columns(4)
records.metric("Records", f"{len(frame):,}")
features.metric("Features", frame.shape[1] - 1)
positives.metric("Positive rate", f"{positive_rate:.1%}")
complete.metric("Cells complete", f"{frame.notna().to_numpy().mean():.1%}")

figure(
    "Class balance",
    class_balance_bar(frame, target),
    f"{positive_rate:.1%} of records carry the positive label. A rule that always answers "
    "negative already scores the remaining share, which is why accuracy leads nothing on this page.",
)

composition, history = st.columns(2)
with composition:
    figure(
        "Gender composition",
        category_donut(frame, "gender"),
        "Three reported values. A class small enough to be unusable in a stratified split is a "
        "modelling constraint, not a curiosity.",
    )
with history:
    figure(
        "Smoking history",
        category_bar(frame, "smoking_history", highlight="No Info"),
        "No Info is missingness recorded as a category. It is highlighted because cleaning has to "
        "decide whether it becomes a null, its own level, or a dropped row.",
    )

st.subheader("Project status")
st.table(
    [
        {"Stage": "Milestone 1, planning", "Owner": "All", "Status": "Submitted"},
        {"Stage": "Milestone 2, data engineering", "Owner": "Axelle", "Status": "In progress"},
        {"Stage": "Baselines and hybrid model", "Owner": "Irfan", "Status": "Not started"},
        {"Stage": "Dashboard", "Owner": "Pison", "Status": "Shell with dummy data"},
    ]
)

st.subheader("Dataset of record")
config = dataset_config()
st.write(config["candidates"][config["active"]])

disclaimer()

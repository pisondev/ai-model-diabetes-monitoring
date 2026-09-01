import streamlit as st

from lib.config import APP_SUBTITLE, APP_TITLE, dataset_config, target_name
from lib.data import load_dataset
from lib.ui import disclaimer, page, source_badge

page(APP_TITLE, APP_SUBTITLE)

frame, source = load_dataset()
source_badge(source)

left, middle, right = st.columns(3)
left.metric("Records", f"{len(frame):,}")
middle.metric("Features", frame.shape[1] - 1)
right.metric("Positive rate", f"{frame[target_name()].mean():.1%}")

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

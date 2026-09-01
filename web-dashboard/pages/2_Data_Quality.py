import streamlit as st

from lib.config import target_name
from lib.data import load_dataset
from lib.ui import page, source_badge

page("Data quality", "Evidence that supports the cleaning decisions")

frame, source = load_dataset()
source_badge(source)

left, right = st.columns(2)
left.metric("Duplicate rows", int(frame.duplicated().sum()))
right.metric("Cells missing", int(frame.isna().sum().sum()))

st.subheader("Missing values per field")
st.bar_chart(frame.isna().sum())

st.subheader("Class balance")
st.bar_chart(frame[target_name()].value_counts())

st.info(
    "Before and after cleaning evidence is produced in 02_data_engineering_lab.ipynb. "
    "This page reads whatever dataset version is present and stays honest about which one."
)

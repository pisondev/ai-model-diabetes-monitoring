import streamlit as st

from lib.config import schema
from lib.data import load_dataset
from lib.ui import page, source_badge

page("Dataset overview", "Structure profile of the dataset the project runs on")

frame, source = load_dataset()
source_badge(source)

st.subheader("Sample rows")
st.dataframe(frame.head(20))

st.subheader("Field contract")
st.table(
    [
        {"Field": name, "Type": field["type"], "Definition": field.get("definition", "")}
        for name, field in schema()["features"].items()
    ]
)

st.info("Structure and behaviour profiling belongs to the data side and lands in data-engineering/reports/.")

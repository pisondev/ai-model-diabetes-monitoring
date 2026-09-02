import streamlit as st

from lib.charts import histogram, positive_rate_by_band, scatter_by_class
from lib.config import role, schema, target_name
from lib.data import load_dataset
from lib.ui import figure, page, source_badge

page("Patient cohort", "Who the frozen cohort is and where the diagnosis separates")

frame, source = load_dataset()
source_badge(source)

target = target_name()
primary, secondary, body, age = (role(name) for name in ("primary", "secondary", "body", "age"))

left, right = st.columns(2)
with left:
    figure(
        f"{primary} distribution",
        histogram(frame, primary),
        "The measure the diagnosis is defined against, so its shape sets the ceiling any model "
        "can reach on this cohort.",
    )
with right:
    figure(
        f"{body} distribution",
        histogram(frame, body),
        "Cleaning replaced the zero-coded values here, which is why the left tail starts at a "
        "plausible reading rather than at zero.",
    )

figure(
    f"{primary} against {secondary}, coloured by diagnosis",
    scatter_by_class(frame, primary, secondary, target),
    "Two horizontal lines run through this cloud, one per class. They are the outcome-stratified "
    "median the cleaning step imputed into the 374 missing insulin readings, and any model trained "
    "on this column will learn them, so the modeling side has to know they are there.",
)

figure(
    f"Diabetic share by {age} band",
    positive_rate_by_band(frame, age, target),
    "Quantile bands, so each bar rests on about the same number of patients. A share that climbs "
    "with the band is the signal age is worth keeping as a feature.",
)

st.subheader("Cohort sample")
st.dataframe(frame.head(20), use_container_width=True)

st.subheader("Field contract")
st.table(
    [
        {
            "Field": name,
            "Type": field["type"],
            "Unit": field.get("unit", ""),
            "Range": f"{field.get('min', '')} to {field.get('max', '')}",
            "Definition": field.get("definition", ""),
        }
        for name, field in schema()["features"].items()
    ]
)

st.caption(
    "Columns beyond the contract are derived by the cleaning pipeline and documented in "
    "data-engineering/reports/data-dictionary.md."
)

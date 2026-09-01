import streamlit as st

from lib.charts import histogram, positive_rate_by_band, scatter_by_class
from lib.config import schema, target_name
from lib.data import load_dataset
from lib.ui import figure, page, source_badge

page("Dataset overview", "Structure profile of the dataset the project runs on")

frame, source = load_dataset()
source_badge(source)

target = target_name()

age, bmi = st.columns(2)
with age:
    figure(
        "Age distribution",
        histogram(frame, "age"),
        "The shape of the age column decides whether age bands are worth building as a feature.",
    )
with bmi:
    figure(
        "BMI distribution",
        histogram(frame, "bmi"),
        "The declared range reaches 96, far past anything clinically plausible. The right tail is "
        "what the outlier study has to explain.",
    )

figure(
    "HbA1c against blood glucose, coloured by label",
    scatter_by_class(frame, "HbA1c_level", "blood_glucose_level", target),
    "Both axes are close to definitional for the label. A clean separation here is evidence that "
    "the labelling rule is being recovered, not that the problem is easy.",
)

figure(
    "Positive rate by age band",
    positive_rate_by_band(frame, "age", target),
    "Equal-sized bands, so each bar rests on the same number of records. A rate that climbs with "
    "the band is the signal age is worth keeping.",
)

st.subheader("Sample rows")
st.dataframe(frame.head(20), use_container_width=True)

st.subheader("Field contract")
st.table(
    [
        {"Field": name, "Type": field["type"], "Definition": field.get("definition", "")}
        for name, field in schema()["features"].items()
    ]
)

st.info("Structure and behaviour profiling belongs to the data side and lands in data-engineering/reports/.")

import streamlit as st

from lib.charts import cumulative_distribution, record_vs_cohort_dumbbell, risk_meter
from lib.config import numeric_features, schema
from lib.data import load_dataset
from lib.predictor import load_predictor
from lib.ui import disclaimer, figure, page

page("Risk screening", "Single record screening against the current predictor")

frame, _ = load_dataset()
spec = schema()["features"]
predictor = load_predictor()

if predictor.source == "dummy":
    st.warning("No trained model is loaded. Scores below are placeholders, not predictions.")

with st.form("record"):
    record = {}
    columns = st.columns(2)
    for index, (name, field) in enumerate(spec.items()):
        column = columns[index % 2]
        if field["type"] == "categorical":
            record[name] = column.selectbox(name, field["values"])
        elif field["type"] == "binary":
            record[name] = int(column.checkbox(name))
        elif field["type"] == "int":
            record[name] = column.number_input(name, field["min"], field["max"], step=1)
        else:
            record[name] = column.number_input(name, float(field["min"]), float(field["max"]))
    if st.form_submit_button("Screen record"):
        st.session_state["screened"] = record

if "screened" in st.session_state:
    screened = st.session_state["screened"]
    result = predictor.predict(screened)

    score, verdict = st.columns([1, 2])
    score.metric("Risk score", f"{result.probability:.3f}")
    verdict.metric("Verdict", "Positive" if result.label else "Negative")

    figure(
        "Score against the decision threshold",
        risk_meter(result.probability, predictor.threshold),
        f"The dark rule is the threshold at {predictor.threshold:.2f}. Moving it trades recall "
        "against precision, and the choice belongs in the evaluation, not in the interface.",
    )

    distribution, comparison = st.columns(2)
    with distribution:
        figure(
            "Where this HbA1c sits in the cohort",
            cumulative_distribution(frame, "HbA1c_level", marker=screened["HbA1c_level"]),
            "The curve is the dataset, the marked point is the record entered above.",
        )
    with comparison:
        figure(
            "This record against the cohort median",
            record_vs_cohort_dumbbell(screened, frame, {name: spec[name] for name in numeric_features()}),
            "Each field is placed inside its own declared range, so four different units share one axis.",
        )

disclaimer()

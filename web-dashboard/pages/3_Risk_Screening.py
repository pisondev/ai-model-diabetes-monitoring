import streamlit as st

from lib.charts import cumulative_distribution, record_vs_cohort_dumbbell, risk_meter
from lib.config import numeric_features, role, schema
from lib.data import load_dataset
from lib.predictor import load_predictor
from lib.ui import disclaimer, figure, page

page("Risk screening", "Score one patient against the cohort and the current predictor")

frame, _ = load_dataset()
spec = schema()["features"]
predictor = load_predictor()
primary = role("primary")

if predictor.source == "dummy":
    st.warning("No trained model is loaded. Scores below are placeholders, not predictions.")

with st.form("record"):
    record = {}
    columns = st.columns(2)
    for index, (name, field) in enumerate(spec.items()):
        column = columns[index % 2]
        label = f"{name} ({field['unit']})" if field.get("unit") else name
        if field["type"] == "categorical":
            record[name] = column.selectbox(label, field["values"])
        elif field["type"] == "binary":
            record[name] = int(column.checkbox(label))
        elif field["type"] == "int":
            record[name] = column.number_input(label, field["min"], field["max"], step=1)
        else:
            record[name] = column.number_input(label, float(field["min"]), float(field["max"]))
    if st.form_submit_button("Screen patient"):
        st.session_state["screened"] = record

if "screened" in st.session_state:
    screened = st.session_state["screened"]
    result = predictor.predict(screened)
    percentile = float((frame[primary] <= screened[primary]).mean())

    score, verdict, position = st.columns(3)
    score.metric("Risk score", f"{result.probability:.3f}")
    verdict.metric("Screened as", "Diabetic" if result.label else "Not diabetic")
    position.metric(f"{primary} percentile", f"{percentile:.0%}")

    figure(
        "Score against the decision threshold",
        risk_meter(result.probability, predictor.threshold),
        f"The dark rule is the threshold at {predictor.threshold:.2f}. Moving it trades recall "
        "against precision, and that choice belongs in the evaluation rather than in this form.",
    )

    distribution, comparison = st.columns(2)
    with distribution:
        figure(
            f"Where this {primary} sits in the cohort",
            cumulative_distribution(frame, primary, marker=screened[primary]),
            "The curve is the frozen cohort, the marked point is the patient entered above.",
        )
    with comparison:
        figure(
            "This patient against the cohort median",
            record_vs_cohort_dumbbell(screened, frame, {name: spec[name] for name in numeric_features()}),
            "Each reading is placed inside its own declared range, so eight different units share one axis.",
        )

disclaimer()

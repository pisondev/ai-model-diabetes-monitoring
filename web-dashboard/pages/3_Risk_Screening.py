import streamlit as st

from lib.charts import cumulative_distribution, record_vs_cohort_dumbbell, risk_meter
from lib.config import numeric_features, role, schema
from lib.data import load_dataset
from lib.predictor import load_predictor
from lib.ui import disclaimer, figure, page

page("Risk screening", "Score one patient against the cohort and the trained model")

frame, _ = load_dataset()
spec = schema()["features"]
predictor = load_predictor()
primary = role("primary")

# two real patients from the raw download whose readings all sit inside the declared ranges,
# so a demo does not depend on typing eight numbers
EXAMPLES = {
    "Pasien risiko tinggi": {"Pregnancies": 7, "Glucose": 181, "BloodPressure": 84, "SkinThickness": 21,
                             "Insulin": 192, "BMI": 35.9, "DiabetesPedigreeFunction": 0.586, "Age": 51},
    "Pasien risiko rendah": {"Pregnancies": 2, "Glucose": 56, "BloodPressure": 56, "SkinThickness": 28,
                             "Insulin": 45, "BMI": 24.2, "DiabetesPedigreeFunction": 0.332, "Age": 22},
}


def inside(field, value):
    """A preset can only offer a reading the contract allows, so the form never refuses it."""
    return max(field["min"], min(field["max"], value))


if predictor.source == "trained":
    st.success(
        f"Model loaded: {predictor.name}, decision threshold {predictor.threshold:.3f}, "
        f"trained {predictor.trained_at}. Scores below are real predictions."
    )
else:
    st.warning(
        "No trained artifact found. Run `python modeling/train.py` to publish one. Until then the "
        "scores below come from a placeholder."
    )

st.session_state.setdefault("generation", 0)
st.session_state.setdefault("preset", {})
first, second, _ = st.columns([1, 1, 2])
for column, (label, values) in zip((first, second), EXAMPLES.items()):
    if column.button(label, use_container_width=True):
        st.session_state["preset"] = values
        st.session_state["generation"] += 1
        st.rerun()

preset = st.session_state["preset"]
generation = st.session_state["generation"]

with st.form("record"):
    record = {}
    columns = st.columns(2)
    for index, (name, field) in enumerate(spec.items()):
        column = columns[index % 2]
        label = f"{name} ({field['unit']})" if field.get("unit") else name
        key = f"{name}_{generation}"
        if field["type"] == "int":
            default = int(inside(field, preset.get(name, field["min"])))
            record[name] = column.number_input(label, field["min"], field["max"], value=default,
                                               step=1, key=key)
        else:
            default = float(inside(field, preset.get(name, field["min"])))
            record[name] = column.number_input(label, float(field["min"]), float(field["max"]),
                                               value=default, key=key)
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
        f"The dark rule is the threshold at {predictor.threshold:.3f}. It was chosen on "
        "out-of-fold predictions to hold recall above 0.70, because missing a diabetic patient "
        "costs more than a false alarm.",
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

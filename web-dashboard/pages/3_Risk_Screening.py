import streamlit as st

from lib.config import schema
from lib.predictor import load_predictor
from lib.ui import disclaimer, page

page("Risk screening", "Single record screening against the current predictor")

predictor = load_predictor()
if predictor.source == "dummy":
    st.warning("No trained model is loaded. Scores below are placeholders, not predictions.")

record = {}
columns = st.columns(2)
for index, (name, field) in enumerate(schema()["features"].items()):
    column = columns[index % 2]
    if field["type"] == "categorical":
        record[name] = column.selectbox(name, field["values"])
    elif field["type"] == "binary":
        record[name] = int(column.checkbox(name))
    elif field["type"] == "int":
        record[name] = column.number_input(name, field["min"], field["max"], step=1)
    else:
        record[name] = column.number_input(name, float(field["min"]), float(field["max"]))

if st.button("Screen record"):
    result = predictor.predict(record)
    st.metric("Risk score", f"{result.probability:.3f}")
    st.write("Screened as positive" if result.label else "Screened as negative")

disclaimer()

import streamlit as st

from lib.charts import (
    category_donut,
    class_balance_bar,
    ordinal_donut,
    positive_rate_by_band,
    prevalence_line,
)
from lib.config import APP_SUBTITLE, APP_TITLE, role, target_name
from lib.data import derived_column, load_dataset, top_risk_patients
from lib.ui import disclaimer, figure, module_card, page, source_badge

page(APP_TITLE, APP_SUBTITLE)

frame, source = load_dataset()
source_badge(source)

target = target_name()
cases = int(frame[target].sum())
prevalence = frame[target].mean()
score_column = derived_column(frame, "risk_score")
body_band = derived_column(frame, "body_band")

patients, diagnosed, rate, elevated = st.columns(4)
patients.metric("Patients screened", f"{len(frame):,}")
diagnosed.metric("Diabetic cases", f"{cases:,}")
rate.metric("Prevalence", f"{prevalence:.1%}")
if score_column:
    high = int((frame[score_column] >= 2).sum())
    elevated.metric("Elevated metabolic risk", f"{high:,}", help=f"{high / len(frame):.1%} of the cohort scores 2 or more")
else:
    elevated.metric("Cells complete", f"{frame.notna().to_numpy().mean():.1%}")

trend, composition, distribution, critical = st.columns(4)

with trend:
    figure(
        "Prevalence by age",
        prevalence_line(frame, role("age"), target),
        "Quantile age bands. The dark rule is the cohort share.",
    )

with composition:
    if body_band:
        figure(
            "Cases by body-mass band",
            category_donut(frame, body_band),
            "WHO bands assigned by the cleaning pipeline.",
        )
    else:
        figure("Class balance", class_balance_bar(frame, target), "Diagnosed against screened.")

with distribution:
    if score_column:
        figure(
            "Risk score distribution",
            ordinal_donut(frame, score_column),
            "Rule-based score, zero to four, from the derived features.",
        )
    else:
        figure(
            f"Diabetic share by {role('body')} band",
            positive_rate_by_band(frame, role("body"), target),
            "The derived score is not in this frame, so the body measure stands in for it.",
        )

with critical:
    st.markdown("**Highest risk patients**")
    ranked = top_risk_patients(frame, target)
    shown = ["patient", role("primary")] + ([score_column] if score_column else []) + [target]
    table = ranked[shown].rename(
        columns={"patient": "Patient", role("primary"): "Gluc.", score_column or target: "Score", target: "Dx"}
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Ranked by the derived score, then by glucose. Dx is the recorded diagnosis.")

st.subheader("Where to go next")
first, second, third = st.columns(3)
with first:
    module_card(
        "Patient cohort",
        "Distributions, separation and the field contract",
        "pages/1_Patient_Cohort.py",
    )
with second:
    module_card(
        "Risk screening",
        "Score a single patient against the cohort",
        "pages/3_Risk_Screening.py",
    )
with third:
    module_card(
        "Model performance",
        "The floor a trained model has to clear",
        "pages/4_Model_Performance.py",
    )

disclaimer()

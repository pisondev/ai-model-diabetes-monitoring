import json

import streamlit as st

from lib.charts import accuracy_vs_prevalence_line, majority_baseline_bar, positive_rate_by_level
from lib.config import METRICS_FILE, target_name
from lib.data import derived_column, load_dataset
from lib.ui import figure, page, source_badge

page("Model performance", "What the trained model scores, and what it had to clear")

frame, source = load_dataset()
source_badge(source)

target = target_name()
prevalence = frame[target].mean()
score_column = derived_column(frame, "risk_score")
metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8")) if METRICS_FILE.exists() else None

if metrics:
    test = metrics["test"]
    st.success(
        f"{metrics['model']}. Trained {metrics['trained_at']}, seed {metrics['seed']}, "
        f"{metrics['rows']['train']} patients for training and {metrics['rows']['test']} held out."
    )

    pr, recall, f1, roc = st.columns(4)
    pr.metric("PR-AUC", f"{test['pr_auc']:.3f}", f"{test['pr_auc'] - prevalence:+.3f} vs floor")
    recall.metric("Recall", f"{test['recall']:.3f}", f"{test['recall']:+.3f} vs floor")
    f1.metric("F1", f"{test['f1']:.3f}")
    roc.metric("ROC-AUC", f"{test['roc_auc']:.3f}", f"{test['roc_auc'] - 0.5:+.3f} vs coin flip")
    st.caption(
        f"Held-out test set, decision threshold {test['threshold']:.3f}. Precision {test['precision']:.3f}, "
        f"accuracy {test['accuracy']:.3f}. The floor is the rule that always answers negative."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Which learner earned its place")
        board = [
            {"Model": name.replace("_", " "), "CV PR-AUC": f"{row['cv_pr_auc']:.3f}",
             "CV ROC-AUC": f"{row['cv_roc_auc']:.3f}"}
            for name, row in metrics["leaderboard"].items()
        ]
        st.dataframe(board, hide_index=True, use_container_width=True)
        st.caption(
            f"Stratified {metrics['split']['folds']}-fold cross validation on the training split only. "
            "The ensemble is kept because it leads this table, not because stacking sounds better."
        )
    with right:
        st.subheader("Where the errors fall")
        matrix = test["confusion_matrix"]
        st.dataframe(
            [
                {"": "Truly not diabetic", "Predicted not diabetic": matrix[0][0],
                 "Predicted diabetic": matrix[0][1]},
                {"": "Truly diabetic", "Predicted not diabetic": matrix[1][0],
                 "Predicted diabetic": matrix[1][1]},
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"The threshold is tuned for recall, so the model accepts {matrix[0][1]} false alarms to "
            f"miss only {matrix[1][0]} diabetic patients."
        )

    st.info(metrics["leakage_note"].capitalize() + ".")
else:
    st.warning(
        "No metrics file found. Run `python modeling/train.py` to train the model and publish it. "
        "Until then this page shows only the floor a model has to clear."
    )

st.subheader("The floor it had to clear")
floor, warning = st.columns(2)
with floor:
    figure(
        "The always-negative rule scored on this cohort",
        majority_baseline_bar(prevalence),
        "A model that never predicts diabetes. It finds nobody, and it still reports the accuracy "
        "on the left. That number is the one to beat and the one to distrust.",
    )
with warning:
    figure(
        "Why accuracy is the wrong headline",
        accuracy_vs_prevalence_line(prevalence),
        "Accuracy of that same rule as prevalence moves. The marked point is this cohort, which is "
        "why the metrics above lead with PR-AUC and recall.",
    )

if score_column:
    figure(
        "The rule-based score the model also had to beat",
        positive_rate_by_level(frame, score_column, target),
        "Diabetic share at each level of the derived metabolic risk score. Beating the "
        "always-negative rule is not enough; the model has to order patients better than this.",
    )

st.caption(
    "Every number on this page is read from modeling/models/metrics.json, written by the training "
    "run. Nothing here is typed in by hand."
)

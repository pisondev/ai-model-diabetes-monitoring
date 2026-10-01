"""Trains the screening model and writes the artifact the dashboard loads.

    python modeling/train.py

Seeds are fixed, the split is recorded, and the configuration is written next to the metrics,
so rerunning the command reproduces the reported numbers.
"""

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import dataset, models  # noqa: E402

SEED = 42
TEST_SHARE = 0.2
FOLDS = 10
OUT = Path(__file__).resolve().parent / "models"


def scored(y_true, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "confusion_matrix": matrix.tolist(),
        "threshold": float(threshold),
    }


def best_threshold(y_true, probability, floor=0.70):
    """Highest F1 among thresholds that still reach a recall floor, since a missed patient costs more."""
    grid = np.unique(np.round(np.quantile(probability, np.linspace(0.02, 0.98, 97)), 4))
    allowed = [t for t in grid if recall_score(y_true, (probability >= t).astype(int), zero_division=0) >= floor]
    pool = allowed or list(grid)
    return float(max(pool, key=lambda t: f1_score(y_true, (probability >= t).astype(int), zero_division=0)))


def main():
    started = time.time()
    features, target = dataset.load()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SHARE, stratify=target, random_state=SEED)
    weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    print("rows %d, train %d, test %d, positives in train %.1f%%"
          % (len(features), len(x_train), len(x_test), 100 * y_train.mean()))

    leaderboard = {}
    for name, model in models.candidates(weight).items():
        out_of_fold = cross_val_predict(model, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
        leaderboard[name] = {"cv_pr_auc": float(average_precision_score(y_train, out_of_fold)),
                             "cv_roc_auc": float(roc_auc_score(y_train, out_of_fold))}
        print("  %-20s cv PR-AUC %.3f" % (name, leaderboard[name]["cv_pr_auc"]))

    ensemble = models.stacking(weight)
    stack_oof = cross_val_predict(ensemble, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    leaderboard["stacking"] = {"cv_pr_auc": float(average_precision_score(y_train, stack_oof)),
                               "cv_roc_auc": float(roc_auc_score(y_train, stack_oof))}
    print("  %-20s cv PR-AUC %.3f" % ("stacking", leaderboard["stacking"]["cv_pr_auc"]))

    threshold = best_threshold(y_train, stack_oof)
    ensemble.fit(x_train, y_train)
    test_probability = ensemble.predict_proba(x_test)[:, 1]

    prevalence = float(target.mean())
    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "split": {"test_share": TEST_SHARE, "stratified": True, "folds": FOLDS},
        "rows": {"total": int(len(features)), "train": int(len(x_train)), "test": int(len(x_test))},
        "features": list(features.columns),
        "model": "stacking: logistic regression, random forest, xgboost, svm, meta logistic regression",
        "leaderboard": leaderboard,
        "chosen_threshold": threshold,
        "test": scored(y_test, test_probability, threshold),
        "test_at_half": scored(y_test, test_probability, 0.5),
        "baseline_always_negative": {"pr_auc": prevalence, "roc_auc": 0.5, "recall": 0.0,
                                     "accuracy": 1 - prevalence},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "leakage_note": "trained on the raw file with impossible zeros as missing, imputation inside "
                        "the pipeline so it never sees the label",
        "seconds": round(time.time() - started, 1),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": ensemble, "threshold": threshold, "features": list(features.columns),
                 "trained_at": metrics["trained_at"]}, OUT / "screening_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    test = metrics["test"]
    print("\nheld out test, threshold %.3f" % threshold)
    print("  PR-AUC %.3f   recall %.3f   precision %.3f   F1 %.3f   ROC-AUC %.3f   accuracy %.3f"
          % (test["pr_auc"], test["recall"], test["precision"], test["f1"], test["roc_auc"],
             test["accuracy"]))
    print("  baseline that always answers negative: PR-AUC %.3f, recall 0.000" % prevalence)
    print("\nwritten to %s" % OUT)


if __name__ == "__main__":
    main()

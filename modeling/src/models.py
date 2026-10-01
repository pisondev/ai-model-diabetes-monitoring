"""Candidate learners and the stacking ensemble, each wrapped so preprocessing cannot leak.

Every candidate is a Pipeline. Imputation and scaling therefore fit on the training part of
whatever split surrounds them, never on the whole frame, which is what keeps cross validation
and the held out test honest.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42


def _impute():
    return ("impute", SimpleImputer(strategy="median"))


def logistic_regression():
    return Pipeline([_impute(), ("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=SEED))])


def random_forest():
    return Pipeline([_impute(),
                     ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                      class_weight="balanced_subsample",
                                                      random_state=SEED, n_jobs=-1))])


def gradient_boosting(positive_weight=1.0):
    return Pipeline([_impute(),
                     ("model", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3,
                                             subsample=0.9, colsample_bytree=0.9,
                                             scale_pos_weight=positive_weight, eval_metric="logloss",
                                             random_state=SEED, n_jobs=-1))])


def support_vector():
    # calibrated rather than SVC(probability=True), which scikit-learn 1.9 deprecates
    return Pipeline([_impute(), ("scale", StandardScaler()),
                     ("model", CalibratedClassifierCV(
                         SVC(C=2.0, class_weight="balanced", random_state=SEED),
                         ensemble=False, cv=3))])


def candidates(positive_weight=1.0):
    """The single learners, in the order the presentation lists them."""
    return {
        "logistic_regression": logistic_regression(),
        "random_forest": random_forest(),
        "xgboost": gradient_boosting(positive_weight),
        "svm": support_vector(),
    }


def stacking(positive_weight=1.0, folds=5):
    """Level-0 learners feed out-of-fold predictions to a deliberately simple level-1 model."""
    return StackingClassifier(
        estimators=[(name, model) for name, model in candidates(positive_weight).items()],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        cv=folds,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )

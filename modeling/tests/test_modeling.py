import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src import dataset, models


def test_impossible_zeros_become_missing_and_valid_ones_survive():
    features, target = dataset.load()
    assert len(features) == len(target) == 768
    for column in dataset.IMPOSSIBLE_ZERO:
        assert (features[column] == 0).sum() == 0
        assert features[column].isna().sum() > 0
    assert (features["Pregnancies"] == 0).sum() > 0
    assert features["Pregnancies"].isna().sum() == 0


def test_the_feature_order_follows_the_contract():
    features, _ = dataset.load()
    assert list(features.columns) == dataset.feature_names()


def test_every_candidate_carries_its_own_imputer():
    for name, model in models.candidates().items():
        assert isinstance(model, Pipeline), name
        assert "impute" in dict(model.steps), name


def test_the_stack_cannot_see_the_label_through_preprocessing():
    """A fold with a column missing in training must still score, which only works when the
    imputer is fitted inside the fold rather than on the whole frame."""
    rows = 120
    generator = np.random.default_rng(0)
    frame = pd.DataFrame({name: generator.normal(size=rows) for name in ("a", "b", "c")})
    frame.loc[: rows // 2, "a"] = np.nan
    target = pd.Series((generator.random(rows) > 0.6).astype(int))
    model = models.logistic_regression()
    scores = cross_val_score(model, frame, target, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                             scoring="average_precision")
    assert len(scores) == 3
    assert not np.isnan(scores).any()


def test_the_threshold_search_prefers_recall_over_a_tidy_f1():
    from train import best_threshold

    truth = np.array([0] * 80 + [1] * 20)
    probability = np.concatenate([np.linspace(0.0, 0.6, 80), np.linspace(0.3, 0.95, 20)])
    threshold = best_threshold(truth, probability, floor=0.9)
    predicted = (probability >= threshold).astype(int)
    assert predicted[truth == 1].mean() >= 0.9


@pytest.mark.parametrize("builder", ["random_forest", "xgboost", "svm", "logistic_regression"])
def test_candidates_fit_and_predict_a_probability(builder):
    features, target = dataset.load()
    model = models.candidates()[builder]
    model.fit(features.head(200), target.head(200))
    probability = model.predict_proba(features.head(5))[:, 1]
    assert ((probability >= 0) & (probability <= 1)).all()

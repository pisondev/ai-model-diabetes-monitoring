import hashlib
from dataclasses import dataclass

import pandas as pd

from .config import MODEL_FILE

THRESHOLD = 0.5


@dataclass
class Prediction:
    probability: float
    label: int
    source: str


class DummyPredictor:
    """Deterministic stand-in, used only when no trained artifact is on disk."""

    source = "dummy"
    name = "placeholder"

    def __init__(self, threshold=THRESHOLD):
        self.threshold = threshold

    def predict(self, record):
        digest = hashlib.sha256(repr(sorted(record.items())).encode()).digest()
        probability = int.from_bytes(digest[:4], "big") / 2**32
        return Prediction(probability, int(probability >= self.threshold), self.source)


class TrainedPredictor:
    """The stacking ensemble written by modeling/train.py, loaded straight from its artifact."""

    source = "trained"

    def __init__(self, bundle):
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.features = list(bundle["features"])
        self.trained_at = bundle.get("trained_at", "unknown")
        self.name = bundle.get("name", "stacking ensemble")

    def predict(self, record):
        row = pd.DataFrame([[record[name] for name in self.features]], columns=self.features)
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        return Prediction(probability, int(probability >= self.threshold), self.source)


def load_predictor():
    """The trained artifact when modeling has published one, the placeholder otherwise."""
    if MODEL_FILE.exists():
        import joblib

        return TrainedPredictor(joblib.load(MODEL_FILE))
    return DummyPredictor()

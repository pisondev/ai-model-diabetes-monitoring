import hashlib
from dataclasses import dataclass

THRESHOLD = 0.5


@dataclass
class Prediction:
    probability: float
    label: int
    source: str


class DummyPredictor:
    """Deterministic stand-in so the dashboard can be built before a model exists."""

    source = "dummy"

    def __init__(self, threshold=THRESHOLD):
        self.threshold = threshold

    def predict(self, record):
        digest = hashlib.sha256(repr(sorted(record.items())).encode()).digest()
        probability = int.from_bytes(digest[:4], "big") / 2**32
        return Prediction(probability, int(probability >= self.threshold), self.source)


def load_predictor():
    # swapped for the trained artifact once the modeling side publishes one
    return DummyPredictor()

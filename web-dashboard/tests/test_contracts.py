from lib.config import feature_names, schema, target_name
from lib.data import make_dummy_frame
from lib.predictor import DummyPredictor


def test_dummy_frame_follows_the_schema_columns():
    frame = make_dummy_frame(rows=50)
    assert list(frame.columns) == feature_names() + [target_name()]


def test_dummy_frame_is_reproducible():
    assert make_dummy_frame(rows=20).equals(make_dummy_frame(rows=20))


def test_categorical_values_stay_inside_the_contract():
    frame = make_dummy_frame(rows=200)
    for name, field in schema()["features"].items():
        if field["type"] == "categorical":
            assert set(frame[name]) <= set(field["values"])


def test_numeric_values_stay_inside_the_declared_range():
    frame = make_dummy_frame(rows=200)
    for name, field in schema()["features"].items():
        if field["type"] in {"int", "float"}:
            assert frame[name].min() >= field["min"]
            assert frame[name].max() <= field["max"]


def test_predictor_is_deterministic_for_one_record():
    record = {"age": 45, "bmi": 28.1, "HbA1c_level": 6.2}
    predictor = DummyPredictor()
    assert predictor.predict(record) == predictor.predict(record)

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


def test_imputation_spikes_flag_a_value_that_carries_the_label():
    from lib.config import numeric_features
    from lib.data import imputation_spikes

    frame = make_dummy_frame(rows=400)
    target = target_name()
    column = numeric_features()[0]
    planted = frame.copy()
    planted.loc[planted.index[:120], column] = 999.0
    planted.loc[planted.index[:120], target] = 1
    hits = imputation_spikes(planted, target)
    assert any(row["field"] == column and row["value"] == 999.0 for row in hits)
    assert not imputation_spikes(frame.assign(**{target: 1}), target)

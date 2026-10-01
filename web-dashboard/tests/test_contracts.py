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


def test_the_trained_artifact_is_used_when_it_exists():
    from lib.config import MODEL_FILE
    from lib.predictor import DummyPredictor, load_predictor

    predictor = load_predictor()
    if MODEL_FILE.exists():
        assert predictor.source == "trained"
        assert 0.0 < predictor.threshold < 1.0
        assert predictor.features == feature_names()
    else:
        assert isinstance(predictor, DummyPredictor)


def test_a_trained_predictor_returns_a_probability_and_a_label_that_agree():
    from lib.config import MODEL_FILE, schema
    from lib.predictor import load_predictor

    if not MODEL_FILE.exists():
        return
    predictor = load_predictor()
    fields = schema()["features"]
    record = {name: float(field["min"]) for name, field in fields.items()}
    result = predictor.predict(record)
    assert 0.0 <= result.probability <= 1.0
    assert result.label == int(result.probability >= predictor.threshold)


def test_every_demo_preset_sits_inside_the_contract():
    import ast
    from pathlib import Path

    from lib.config import schema

    source = Path(__file__).resolve().parents[1] / "pages" / "3_Risk_Screening.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    presets = next(node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Assign)
                   and any(getattr(t, "id", "") == "EXAMPLES" for t in node.targets))
    fields = schema()["features"]
    for key, value in zip(presets.keys, presets.values):
        record = ast.literal_eval(value)
        assert set(record) == set(fields), key.value
        for name, reading in record.items():
            assert fields[name]["min"] <= reading <= fields[name]["max"], (key.value, name, reading)

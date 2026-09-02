import numpy as np
import pandas as pd
import pytest

from lib import charts, palette
from lib.config import categorical_features, numeric_features, role, schema, target_name
from lib.data import make_dummy_frame

FRAME = make_dummy_frame(rows=400)
TARGET = target_name()
NUMERIC = numeric_features()
SPEC = schema()["features"]
RECORD = {name: float(FRAME[name].iloc[0]) for name in NUMERIC}
PRIMARY, SECONDARY, BODY, AGE = (role(name) for name in ("primary", "secondary", "body", "age"))

# stands in for the bands the cleaning pipeline derives, so the banded charts stay covered
# whether or not the engineered cohort is on disk
BANDED = FRAME.assign(
    band=pd.cut(FRAME[BODY], 3, labels=["low", "middle", "high"]).astype(str),
    score=(FRAME[PRIMARY] > FRAME[PRIMARY].median()).astype(int)
    + (FRAME[BODY] > FRAME[BODY].median()).astype(int)
    + (FRAME[AGE] > FRAME[AGE].median()).astype(int),
)


def datasets(chart):
    """Every inline dataset a compiled chart carries, largest first."""
    spec = chart.to_dict()
    found = list(spec.get("datasets", {}).values())
    found.extend(_inline(spec))
    return sorted((rows for rows in found if isinstance(rows, list)), key=len, reverse=True)


def _inline(node):
    if isinstance(node, dict):
        if isinstance(node.get("values"), list):
            yield node["values"]
        for key, value in node.items():
            if key != "datasets":
                yield from _inline(value)
    elif isinstance(node, list):
        for value in node:
            yield from _inline(value)


def marks(chart):
    return set(_marks(chart.to_dict()))


def _marks(node):
    if isinstance(node, dict):
        if "mark" in node:
            mark = node["mark"]
            yield mark if isinstance(mark, str) else mark.get("type")
        for value in node.values():
            yield from _marks(value)
    elif isinstance(node, list):
        for value in node:
            yield from _marks(value)


def _colour_encodings(node):
    if isinstance(node, dict):
        if "color" in node and isinstance(node["color"], dict):
            yield node["color"]
        for value in node.values():
            yield from _colour_encodings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _colour_encodings(value)


def scale_of(chart):
    return next(
        encoding["scale"] for encoding in _colour_encodings(chart.to_dict()) if "scale" in encoding
    )


BUILDERS = {
    "class_balance": lambda: charts.class_balance_bar(FRAME, TARGET),
    "donut": lambda: charts.category_donut(BANDED, "band"),
    "ordinal_donut": lambda: charts.ordinal_donut(BANDED, "score"),
    "histogram": lambda: charts.histogram(FRAME, PRIMARY),
    "category_bar": lambda: charts.category_bar(BANDED, "band", highlight="high"),
    "scatter": lambda: charts.scatter_by_class(FRAME, PRIMARY, SECONDARY, TARGET),
    "completeness": lambda: charts.completeness_bar(FRAME),
    "boxplots": lambda: charts.numeric_boxplots(FRAME, NUMERIC),
    "heatmap": lambda: charts.correlation_heatmap(FRAME, NUMERIC + [TARGET]),
    "rate_by_category": lambda: charts.positive_rate_by_category(BANDED, "band", TARGET),
    "rate_by_level": lambda: charts.positive_rate_by_level(BANDED, "score", TARGET),
    "rate_by_band": lambda: charts.positive_rate_by_band(FRAME, AGE, TARGET),
    "prevalence_line": lambda: charts.prevalence_line(FRAME, AGE, TARGET),
    "cdf": lambda: charts.cumulative_distribution(FRAME, BODY, marker=30.0),
    "meter": lambda: charts.risk_meter(0.72),
    "dumbbell": lambda: charts.record_vs_cohort_dumbbell(RECORD, FRAME, {n: SPEC[n] for n in NUMERIC}),
    "baseline": lambda: charts.majority_baseline_bar(0.35),
    "prevalence": lambda: charts.accuracy_vs_prevalence_line(0.35),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_builder_compiles_to_a_valid_spec(name):
    assert BUILDERS[name]().to_dict()


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_no_chart_ships_more_rows_than_the_scatter_cap(name):
    for dataset in datasets(BUILDERS[name]()):
        assert len(dataset) <= charts.SCATTER_SAMPLE


@pytest.mark.parametrize("name", sorted(set(BUILDERS) - {"scatter"}))
def test_every_chart_but_the_scatter_aggregates_before_it_leaves_python(name):
    for dataset in datasets(BUILDERS[name]()):
        assert len(dataset) < len(FRAME)


def test_the_form_of_each_chart_matches_the_job_it_does():
    assert "arc" in marks(charts.category_donut(BANDED, "band"))
    assert "arc" in marks(charts.ordinal_donut(BANDED, "score"))
    assert "rect" in marks(charts.correlation_heatmap(FRAME, NUMERIC))
    assert "line" in marks(charts.accuracy_vs_prevalence_line(0.35))
    assert "line" in marks(charts.prevalence_line(FRAME, AGE, TARGET))
    assert "area" in marks(charts.cumulative_distribution(FRAME, BODY))
    assert "circle" in marks(charts.scatter_by_class(FRAME, BODY, AGE, TARGET))


def test_the_two_diagnosis_classes_keep_the_same_colours_everywhere():
    for chart in (
        charts.class_balance_bar(FRAME, TARGET),
        charts.scatter_by_class(FRAME, BODY, AGE, TARGET),
    ):
        found = [
            encoding["scale"]["range"]
            for encoding in _colour_encodings(chart.to_dict())
            if "scale" in encoding and "range" in encoding["scale"]
        ]
        assert list(palette.CLASS_RANGE) in found


def test_a_donut_folds_its_tail_rather_than_growing_more_colours():
    chart = charts.category_donut(BANDED, "band", max_slices=2)
    slices = datasets(chart)[0]
    names = [row["band"] for row in slices]
    assert len(names) == 2
    assert names[-1] == "Other"
    assert sum(row["records"] for row in slices) == len(BANDED)


def test_only_a_folded_tail_gets_the_grey_slice():
    assert scale_of(charts.category_donut(BANDED, "band"))["range"] == list(palette.SERIES[:3])
    assert scale_of(charts.category_donut(BANDED, "band", max_slices=2))["range"][-1] == palette.DEEMPHASIS


def test_an_ordinal_donut_keeps_the_levels_in_order_on_the_single_hue_ramp():
    chart = charts.ordinal_donut(BANDED, "score")
    scale = scale_of(chart)
    assert scale["domain"] == sorted(scale["domain"])
    assert scale["range"] == list(palette.ORDINAL[: len(scale["domain"])])


def test_levels_past_the_ramp_fold_into_a_top_level():
    wide = FRAME.assign(score=np.arange(len(FRAME)) % (len(palette.ORDINAL) + 3))
    order, labels = charts._ordinal_levels(wide, "score")
    assert len(order) == len(palette.ORDINAL)
    assert order[-1].endswith("or more")
    assert set(labels) == set(order)


def test_a_histogram_sends_bins_rather_than_rows():
    bins = 16
    rows = datasets(charts.histogram(FRAME, AGE, bins=bins))[0]
    assert len(rows) == bins
    assert sum(row["records"] for row in rows) == FRAME[AGE].notna().sum()


def test_a_histogram_bar_is_anchored_to_the_baseline():
    assert charts.histogram(FRAME, AGE).to_dict()["encoding"]["y2"] == {"datum": 0}


def test_the_scatter_caps_its_rows_and_stays_reproducible():
    small = charts.scatter_by_class(FRAME, BODY, AGE, TARGET, sample=50)
    again = charts.scatter_by_class(FRAME, BODY, AGE, TARGET, sample=50)
    rows = datasets(small)[0]
    assert len(rows) == 50
    assert rows == datasets(again)[0]


def test_the_box_summary_matches_numpy():
    values = FRAME[BODY].to_numpy()
    stats = charts._box_stats(values)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    assert (stats["q1"], stats["median"], stats["q3"]) == (q1, median, q3)
    assert stats["lower"] >= values.min()
    assert stats["upper"] <= values.max()


def test_the_correlation_scale_diverges_around_a_neutral_middle():
    scale = scale_of(charts.correlation_heatmap(FRAME, NUMERIC))
    assert scale["domain"] == [-1, 0, 1]
    assert scale["range"] == list(palette.DIVERGING)


def test_bands_cover_the_cohort_evenly_and_use_the_ordinal_ramp():
    bands = 4
    chart = charts.positive_rate_by_band(FRAME, AGE, TARGET, bands=bands)
    rows = datasets(chart)[0]
    assert len(rows) == bands
    assert sum(row["records"] for row in rows) == len(FRAME)
    # quantile bands, so ties on an integer field push a band a little off the even share
    even = len(FRAME) / bands
    assert all(abs(row["records"] - even) < 0.2 * even for row in rows)
    assert scale_of(chart)["range"] == list(palette.ORDINAL[:bands])


def test_the_prevalence_line_sends_one_point_per_band_and_stays_a_share():
    bands = 6
    rows = datasets(charts.prevalence_line(FRAME, AGE, TARGET, bands=bands))[0]
    assert len(rows) == bands
    assert sum(row["patients"] for row in rows) == len(FRAME)
    assert all(0.0 <= row["positive_rate"] <= 1.0 for row in rows)


def test_the_cumulative_curve_never_climbs_past_the_whole_cohort():
    chart = charts.cumulative_distribution(FRAME, PRIMARY)
    rows = datasets(chart)[0]
    assert max(row["share"] for row in rows) == 1.0
    assert len({row[PRIMARY] for row in rows}) == len(rows)
    assert chart.to_dict()["encoding"]["y"]["stack"] is None


def test_the_majority_baseline_reports_the_scores_that_rule_actually_earns():
    rate = 0.349
    rows = {row["metric"]: row["score"] for row in datasets(charts.majority_baseline_bar(rate))[0]}
    assert rows["recall"] == 0.0
    assert rows["accuracy"] == pytest.approx(1 - rate)
    assert rows["PR-AUC"] == pytest.approx(rate)
    assert rows["ROC-AUC"] == 0.5


def test_the_meter_changes_colour_on_the_side_of_the_threshold():
    below = charts.risk_meter(0.20, threshold=0.5).to_dict()
    above = charts.risk_meter(0.80, threshold=0.5).to_dict()
    assert palette.ACCENT in str(below)
    assert palette.EMPHASIS in str(above)
    assert palette.EMPHASIS not in str(below)


def test_the_dumbbell_places_every_clinical_field_inside_its_declared_range():
    chart = charts.record_vs_cohort_dumbbell(RECORD, FRAME, {n: SPEC[n] for n in NUMERIC})
    rows = datasets(chart)[0]
    assert len(rows) == 2 * len(NUMERIC)
    assert all(0.0 <= row["position"] <= 1.0 for row in rows)
    assert chart.to_dict()["height"] >= 34 * len(NUMERIC)


def test_the_schema_helpers_split_the_contract_the_way_the_pages_expect():
    assert set(NUMERIC) | set(categorical_features()) <= set(SPEC)
    assert all(SPEC[name]["type"] in {"int", "float"} for name in NUMERIC)
    assert all(SPEC[name]["type"] == "categorical" for name in categorical_features())


def test_every_share_chart_reads_against_the_same_pinned_axis():
    for chart in (
        charts.positive_rate_by_level(BANDED, "score", TARGET),
        charts.prevalence_line(FRAME, AGE, TARGET),
    ):
        spec = chart.to_dict()
        layers = spec.get("layer", [spec])
        domains = [
            layer["encoding"]["y"]["scale"]["domain"]
            for layer in layers
            if "scale" in layer.get("encoding", {}).get("y", {})
        ]
        assert domains and all(domain == [0, 1] for domain in domains)

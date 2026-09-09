"""Chart builders. Every one aggregates in pandas first, so the browser never receives a raw dataset."""

import altair as alt
import numpy as np
import pandas as pd

from . import palette

HEIGHT = 260
SCATTER_SAMPLE = 4000
SCATTER_SEED = 7
CDF_POINTS = 200


def _styled(chart, height=HEIGHT):
    if height is not None:
        chart = chart.properties(height=height)
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            gridColor=palette.GRID,
            gridWidth=1,
            domainColor=palette.AXIS,
            tickColor=palette.AXIS,
            labelColor=palette.INK_MUTED,
            titleColor=palette.INK_SECONDARY,
            titleFontWeight="normal",
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            labelColor=palette.INK_SECONDARY,
            titleColor=palette.INK_SECONDARY,
            titleFontWeight="normal",
            labelFontSize=11,
            titleFontSize=11,
            symbolType="square",
        )
        .configure_header(labelColor=palette.INK_SECONDARY, labelFontSize=11, titleColor=palette.INK_SECONDARY)
    )


def _class_column(frame, target):
    return np.where(frame[target] == 1, palette.POSITIVE_LABEL, palette.NEGATIVE_LABEL)


def class_balance_bar(frame, target):
    counts = (
        pd.Series(_class_column(frame, target), name="class")
        .value_counts()
        .rename_axis("class")
        .reset_index(name="records")
    )
    counts["share"] = counts["records"] / counts["records"].sum()
    base = alt.Chart(counts).encode(
        y=alt.Y("class:N", title=None, sort=[palette.NEGATIVE_LABEL, palette.POSITIVE_LABEL]),
        x=alt.X("records:Q", title="records", axis=alt.Axis(tickCount=6)),
        tooltip=[
            alt.Tooltip("class:N", title="class"),
            alt.Tooltip("records:Q", title="records", format=","),
            alt.Tooltip("share:Q", title="share", format=".1%"),
        ],
    )
    bars = base.mark_bar(height=34, cornerRadiusEnd=4).encode(
        color=alt.Color(
            "class:N",
            scale=alt.Scale(
                domain=[palette.NEGATIVE_LABEL, palette.POSITIVE_LABEL],
                range=list(palette.CLASS_RANGE),
            ),
            legend=None,
        )
    )
    labels = base.mark_text(align="left", dx=6, fontSize=11, color=palette.INK_SECONDARY).encode(
        text=alt.Text("share:Q", format=".1%")
    )
    return _styled(alt.layer(bars, labels), height=150)


def category_donut(frame, column, max_slices=3):
    counts = frame[column].value_counts().rename_axis(column).reset_index(name="records")
    folded = len(counts) > max_slices
    if folded:
        tail = counts["records"].iloc[max_slices - 1 :].sum()
        counts = pd.concat(
            [counts.head(max_slices - 1), pd.DataFrame([{column: "Other", "records": tail}])],
            ignore_index=True,
        )
    counts["share"] = counts["records"] / counts["records"].sum()
    slices = list(counts[column])
    colours = list(palette.SERIES[: len(slices)])
    if folded:
        colours[-1] = palette.DEEMPHASIS
    base = alt.Chart(counts).encode(
        theta=alt.Theta("records:Q", stack=True),
        order=alt.Order("records:Q", sort="descending"),
        color=alt.Color(
            f"{column}:N",
            scale=alt.Scale(domain=slices, range=colours),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip(f"{column}:N", title=column),
            alt.Tooltip("records:Q", title="records", format=","),
            alt.Tooltip("share:Q", title="share", format=".1%"),
        ],
    )
    ring = base.mark_arc(innerRadius=46, outerRadius=84, padAngle=0.012, stroke=palette.SURFACE, strokeWidth=2)
    labels = base.mark_text(radius=100, fontSize=11, color=palette.INK_SECONDARY).encode(
        text=alt.Text("share:Q", format=".0%")
    )
    return _styled(alt.layer(ring, labels), height=270)


def histogram(frame, column, bins=28):
    counts, edges = np.histogram(frame[column].dropna(), bins=bins)
    data = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "records": counts})
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, color=palette.ACCENT, stroke=palette.SURFACE, strokeWidth=1)
        .encode(
            x=alt.X("start:Q", title=column, scale=alt.Scale(nice=False, zero=False), axis=alt.Axis(grid=False, tickCount=8)),
            x2="end:Q",
            y=alt.Y("records:Q", title="records"),
            y2=alt.datum(0),
            tooltip=[
                alt.Tooltip("start:Q", title="from", format=".1f"),
                alt.Tooltip("end:Q", title="to", format=".1f"),
                alt.Tooltip("records:Q", title="records", format=","),
            ],
        )
    )
    return _styled(chart)


def category_bar(frame, column, highlight=None):
    counts = frame[column].value_counts().rename_axis(column).reset_index(name="records")
    counts["share"] = counts["records"] / counts["records"].sum()
    if highlight is None:
        colour = alt.value(palette.ACCENT)
    else:
        colour = alt.condition(
            alt.datum[column] == highlight, alt.value(palette.EMPHASIS), alt.value(palette.DEEMPHASIS)
        )
    chart = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=3, height=22)
        .encode(
            y=alt.Y(f"{column}:N", sort="-x", title=None),
            x=alt.X("records:Q", title="records"),
            color=colour,
            tooltip=[
                alt.Tooltip(f"{column}:N", title=column),
                alt.Tooltip("records:Q", title="records", format=","),
                alt.Tooltip("share:Q", title="share", format=".1%"),
            ],
        )
    )
    return _styled(chart)


def scatter_by_class(frame, x, y, target, sample=SCATTER_SAMPLE, seed=SCATTER_SEED):
    data = frame if len(frame) <= sample else frame.sample(sample, random_state=seed)
    data = data.assign(**{"class": _class_column(data, target)})
    chart = (
        alt.Chart(data[[x, y, "class"]])
        .mark_circle(size=42, opacity=0.55, stroke=palette.SURFACE, strokeWidth=1)
        .encode(
            x=alt.X(f"{x}:Q", scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=8)),
            y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=6)),
            color=alt.Color(
                "class:N",
                scale=alt.Scale(
                    domain=[palette.NEGATIVE_LABEL, palette.POSITIVE_LABEL],
                    range=list(palette.CLASS_RANGE),
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q"), alt.Tooltip("class:N", title="class")],
        )
    )
    return _styled(chart, height=320)


def completeness_bar(frame):
    data = (frame.notna().sum() / len(frame)).rename_axis("field").reset_index(name="complete")
    base = alt.Chart(data).encode(
        y=alt.Y("field:N", sort="x", title=None),
        x=alt.X(
            "complete:Q",
            title="complete",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", tickCount=5),
        ),
        tooltip=[alt.Tooltip("field:N", title="field"), alt.Tooltip("complete:Q", title="complete", format=".2%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, height=18, color=palette.ACCENT)
    labels = base.mark_text(align="right", dx=-6, fontSize=10, color=palette.SURFACE).encode(
        text=alt.Text("complete:Q", format=".1%")
    )
    return _styled(alt.layer(bars, labels), height=280)


def _box_stats(values):
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    reach = 1.5 * (q3 - q1)
    inside = values[(values >= q1 - reach) & (values <= q3 + reach)]
    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "lower": inside.min() if len(inside) else q1,
        "upper": inside.max() if len(inside) else q3,
        "outliers": int(len(values) - len(inside)),
        "min": values.min(),
        "max": values.max(),
    }


def numeric_boxplots(frame, columns, height=300, width=104):
    panels = []
    for column in columns:
        data = pd.DataFrame([{"field": column, **_box_stats(frame[column].dropna().to_numpy())}])
        base = alt.Chart(data).encode(
            x=alt.X("field:N", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)),
            tooltip=[
                alt.Tooltip("median:Q", title="median", format=".2f"),
                alt.Tooltip("q1:Q", title="q1", format=".2f"),
                alt.Tooltip("q3:Q", title="q3", format=".2f"),
                alt.Tooltip("min:Q", title="min", format=".2f"),
                alt.Tooltip("max:Q", title="max", format=".2f"),
                alt.Tooltip("outliers:Q", title="beyond 1.5 IQR", format=","),
            ],
        )
        whisker = base.mark_rule(color=palette.AXIS, strokeWidth=1).encode(
            y=alt.Y("lower:Q", title=None), y2="upper:Q"
        )
        box = base.mark_bar(size=30, cornerRadius=3, color=palette.ACCENT).encode(y="q1:Q", y2="q3:Q")
        middle = base.mark_tick(size=30, thickness=2, color=palette.SURFACE).encode(y="median:Q")
        panels.append(alt.layer(whisker, box, middle).properties(width=width, height=height, title=column))
    concat = alt.hconcat(*panels, spacing=20).resolve_scale(y="independent")
    return _styled(concat, height=None).configure_title(
        fontSize=11, fontWeight="normal", color=palette.INK_SECONDARY, anchor="middle"
    )


def correlation_heatmap(frame, columns):
    matrix = frame[list(columns)].corr(numeric_only=True).round(2)
    long = (
        matrix.reset_index()
        .melt(id_vars="index", var_name="column", value_name="correlation")
        .rename(columns={"index": "row"})
    )
    base = alt.Chart(long).encode(
        x=alt.X("column:N", title=None, sort=list(columns), axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("row:N", title=None, sort=list(columns)),
        tooltip=[
            alt.Tooltip("row:N", title="row"),
            alt.Tooltip("column:N", title="column"),
            alt.Tooltip("correlation:Q", title="r", format=".2f"),
        ],
    )
    cells = base.mark_rect(stroke=palette.SURFACE, strokeWidth=2).encode(
        color=alt.Color(
            "correlation:Q",
            scale=alt.Scale(domain=[-1, 0, 1], range=list(palette.DIVERGING)),
            legend=alt.Legend(title="r", gradientLength=150),
        )
    )
    labels = base.mark_text(fontSize=10).encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.correlation) > 0.55", alt.value(palette.SURFACE), alt.value(palette.INK_SECONDARY)
        ),
    )
    return _styled(alt.layer(cells, labels), height=340)


def positive_rate_by_category(frame, column, target):
    grouped = frame.groupby(column, dropna=False)[target].agg(["mean", "size"]).reset_index()
    grouped.columns = [column, "positive_rate", "records"]
    bars = (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusEnd=3, height=20, color=palette.ACCENT)
        .encode(
            y=alt.Y(f"{column}:N", sort="-x", title=None),
            x=alt.X("positive_rate:Q", title="positive rate", axis=alt.Axis(format="%", tickCount=5)),
            tooltip=[
                alt.Tooltip(f"{column}:N", title=column),
                alt.Tooltip("positive_rate:Q", title="positive rate", format=".1%"),
                alt.Tooltip("records:Q", title="records", format=","),
            ],
        )
    )
    reference = pd.DataFrame({"overall": [frame[target].mean()]})
    rule = (
        alt.Chart(reference)
        .mark_rule(color=palette.INK, strokeWidth=1.5)
        .encode(x="overall:Q", tooltip=alt.Tooltip("overall:Q", title="dataset rate", format=".1%"))
    )
    note = (
        alt.Chart(reference)
        .mark_text(align="left", baseline="top", dx=6, fontSize=10, color=palette.INK_SECONDARY)
        .encode(x="overall:Q", y=alt.value(0), text=alt.value("dataset rate"))
    )
    return _styled(alt.layer(bars, rule, note))


def positive_rate_by_band(frame, column, target, bands=5):
    cut = pd.qcut(frame[column], bands, duplicates="drop")
    grouped = frame.groupby(cut, observed=True)[target].agg(["mean", "size"]).reset_index()
    grouped.columns = ["band", "positive_rate", "records"]
    grouped["band"] = grouped["band"].apply(lambda edge: f"{edge.left:.0f} to {edge.right:.0f}")
    order = list(grouped["band"])
    chart = (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusEnd=3, stroke=palette.SURFACE, strokeWidth=1)
        .encode(
            x=alt.X("band:N", sort=order, title=column, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("positive_rate:Q", title="positive rate", axis=alt.Axis(format="%")),
            color=alt.Color(
                "band:N",
                sort=order,
                scale=alt.Scale(domain=order, range=list(palette.ORDINAL[: len(order)])),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("band:N", title=column),
                alt.Tooltip("positive_rate:Q", title="positive rate", format=".1%"),
                alt.Tooltip("records:Q", title="records", format=","),
            ],
        )
    )
    return _styled(chart)


def cumulative_distribution(frame, column, marker=None, points=CDF_POINTS):
    values = frame[column].dropna().to_numpy()
    quantiles = np.linspace(0, 1, points)
    data = pd.DataFrame({column: np.quantile(values, quantiles), "share": quantiles})
    data = data.groupby(column, as_index=False)["share"].max()
    curve = (
        alt.Chart(data)
        .mark_area(color=palette.ACCENT, opacity=0.12, line={"color": palette.ACCENT, "strokeWidth": 2})
        .encode(
            x=alt.X(f"{column}:Q", scale=alt.Scale(nice=False, zero=False), axis=alt.Axis(grid=False)),
            y=alt.Y(
                "share:Q",
                stack=None,
                title="share of records at or below",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5),
            ),
            tooltip=[alt.Tooltip(f"{column}:Q", format=".2f"), alt.Tooltip("share:Q", format=".1%")],
        )
    )
    if marker is None:
        return _styled(curve)
    here = pd.DataFrame([{column: marker, "share": float((values <= marker).mean())}])
    rule = alt.Chart(here).mark_rule(color=palette.EMPHASIS, strokeWidth=2).encode(x=f"{column}:Q")
    dot = (
        alt.Chart(here)
        .mark_point(filled=True, size=110, color=palette.EMPHASIS, stroke=palette.SURFACE, strokeWidth=2)
        .encode(x=f"{column}:Q", y="share:Q", tooltip=alt.Tooltip("share:Q", title="percentile", format=".1%"))
    )
    label = (
        alt.Chart(here)
        .mark_text(align="left", dx=8, dy=-10, fontSize=11, color=palette.EMPHASIS)
        .encode(x=f"{column}:Q", y="share:Q", text=alt.Text("share:Q", format=".0%"))
    )
    return _styled(alt.layer(curve, rule, dot, label))


def risk_meter(probability, threshold=0.5):
    fill_colour = palette.EMPHASIS if probability >= threshold else palette.ACCENT
    track = (
        alt.Chart(pd.DataFrame({"value": [1.0]}))
        .mark_bar(height=30, cornerRadius=4, color=palette.DEEMPHASIS)
        .encode(
            x=alt.X(
                "value:Q",
                scale=alt.Scale(domain=[0, 1]),
                title=None,
                axis=alt.Axis(format="%", grid=False, tickCount=5),
            )
        )
    )
    fill = (
        alt.Chart(pd.DataFrame({"value": [probability]}))
        .mark_bar(height=30, cornerRadius=4, color=fill_colour)
        .encode(x="value:Q", tooltip=alt.Tooltip("value:Q", title="score", format=".3f"))
    )
    limit = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color=palette.INK, strokeWidth=2)
        .encode(x="threshold:Q", tooltip=alt.Tooltip("threshold:Q", title="threshold", format=".2f"))
    )
    return _styled(alt.layer(track, fill, limit), height=110)


def record_vs_cohort_dumbbell(record, frame, fields):
    rows = []
    for name, spec in fields.items():
        span = spec["max"] - spec["min"]
        pair = (("cohort median", float(frame[name].median())), ("this record", float(record[name])))
        for who, value in pair:
            rows.append({"field": name, "who": who, "position": (value - spec["min"]) / span, "value": value})
    data = pd.DataFrame(rows)
    order = list(fields)
    base = alt.Chart(data).encode(y=alt.Y("field:N", sort=order, title=None))
    connector = base.mark_line(color=palette.DEEMPHASIS, strokeWidth=3).encode(
        x=alt.X(
            "position:Q",
            title="position inside the declared range",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        detail="field:N",
    )
    dots = base.mark_point(filled=True, size=120, stroke=palette.SURFACE, strokeWidth=2).encode(
        x="position:Q",
        color=alt.Color(
            "who:N",
            scale=alt.Scale(domain=["cohort median", "this record"], range=list(palette.SERIES[:2])),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("field:N", title="field"),
            alt.Tooltip("who:N", title="series"),
            alt.Tooltip("value:Q", title="value", format=".2f"),
        ],
    )
    # one row per field, or vega starts dropping every other axis label
    return _styled(alt.layer(connector, dots), height=max(240, 34 * len(order)))


def majority_baseline_bar(positive_rate):
    """Scores the always-negative rule, the floor a trained model has to clear."""
    data = pd.DataFrame(
        [
            {"metric": "accuracy", "score": 1 - positive_rate},
            {"metric": "recall", "score": 0.0},
            {"metric": "PR-AUC", "score": positive_rate},
            {"metric": "ROC-AUC", "score": 0.5},
        ]
    )
    base = alt.Chart(data).encode(
        x=alt.X("metric:N", sort=list(data["metric"]), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("score:Q", title="score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("metric:N", title="metric"), alt.Tooltip("score:Q", title="score", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, size=52, color=palette.ACCENT)
    labels = base.mark_text(dy=-8, fontSize=11, color=palette.INK_SECONDARY).encode(
        text=alt.Text("score:Q", format=".1%")
    )
    return _styled(alt.layer(bars, labels))


def accuracy_vs_prevalence_line(positive_rate, limit=0.5, points=101):
    grid = np.linspace(0.0, limit, points)
    data = pd.DataFrame({"prevalence": grid, "accuracy": 1 - grid})
    curve = (
        alt.Chart(data)
        .mark_line(color=palette.ACCENT, strokeWidth=2)
        .encode(
            x=alt.X(
                "prevalence:Q",
                title="positive rate in the data",
                axis=alt.Axis(format="%", grid=False, tickCount=6),
            ),
            y=alt.Y(
                "accuracy:Q",
                title="accuracy of the always-negative rule",
                scale=alt.Scale(domain=[0.5, 1]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[alt.Tooltip("prevalence:Q", format=".1%"), alt.Tooltip("accuracy:Q", format=".1%")],
        )
    )
    here = pd.DataFrame([{"prevalence": positive_rate, "accuracy": 1 - positive_rate}])
    rule = alt.Chart(here).mark_rule(color=palette.EMPHASIS, strokeWidth=2).encode(x="prevalence:Q")
    dot = (
        alt.Chart(here)
        .mark_point(filled=True, size=120, color=palette.EMPHASIS, stroke=palette.SURFACE, strokeWidth=2)
        .encode(
            x="prevalence:Q",
            y="accuracy:Q",
            tooltip=[
                alt.Tooltip("prevalence:Q", title="this dataset", format=".1%"),
                alt.Tooltip("accuracy:Q", title="accuracy", format=".1%"),
            ],
        )
    )
    label = (
        alt.Chart(here)
        .mark_text(align="left", dx=10, fontSize=11, color=palette.EMPHASIS)
        .encode(x="prevalence:Q", y="accuracy:Q", text=alt.Text("accuracy:Q", format=".1%"))
    )
    return _styled(alt.layer(curve, rule, dot, label))


def _ordinal_levels(frame, column):
    """Sorted levels, with anything past the ramp folded into a top level rather than recoloured."""
    levels = sorted(frame[column].dropna().unique())
    if len(levels) <= len(palette.ORDINAL):
        return [str(level) for level in levels], frame[column].astype(str)
    keep = levels[: len(palette.ORDINAL) - 1]
    top = f"{keep[-1]} or more"
    labels = frame[column].apply(lambda value: str(value) if value in keep else top)
    return [str(level) for level in keep] + [top], labels


def ordinal_donut(frame, column):
    order, labels = _ordinal_levels(frame, column)
    counts = labels.value_counts().rename_axis("level").reset_index(name="patients")
    counts = counts.set_index("level").reindex(order).reset_index()
    counts["share"] = counts["patients"] / counts["patients"].sum()
    base = alt.Chart(counts).encode(
        theta=alt.Theta("patients:Q", stack=True),
        order=alt.Order("level:N", sort="ascending"),
        color=alt.Color(
            "level:N",
            sort=order,
            scale=alt.Scale(domain=order, range=list(palette.ORDINAL[: len(order)])),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("level:N", title="level"),
            alt.Tooltip("patients:Q", title="patients", format=","),
            alt.Tooltip("share:Q", title="share", format=".1%"),
        ],
    )
    ring = base.mark_arc(innerRadius=46, outerRadius=84, padAngle=0.012, stroke=palette.SURFACE, strokeWidth=2)
    labels = base.mark_text(radius=100, fontSize=11, color=palette.INK_SECONDARY).encode(
        text=alt.Text("share:Q", format=".0%")
    )
    return _styled(alt.layer(ring, labels), height=270)


def positive_rate_by_level(frame, column, target, order=None):
    order, labels = (order, frame[column].astype(str)) if order else _ordinal_levels(frame, column)
    grouped = frame.assign(level=labels).groupby("level", observed=True)[target].agg(["mean", "size"]).reset_index()
    grouped.columns = ["level", "positive_rate", "patients"]
    grouped = grouped.set_index("level").reindex(order).reset_index()
    base = alt.Chart(grouped).encode(
        x=alt.X("level:N", sort=order, title=column.replace("_", " "), axis=alt.Axis(labelAngle=0)),
        # pinned so two of these side by side stay comparable at a glance
        y=alt.Y(
            "positive_rate:Q",
            title="diabetic share",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", tickCount=5),
        ),
        tooltip=[
            alt.Tooltip("level:N", title="level"),
            alt.Tooltip("positive_rate:Q", title="diabetic share", format=".1%"),
            alt.Tooltip("patients:Q", title="patients", format=","),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, stroke=palette.SURFACE, strokeWidth=1).encode(
        color=alt.Color(
            "level:N",
            sort=order,
            scale=alt.Scale(domain=order, range=list(palette.ORDINAL[: len(order)])),
            legend=None,
        )
    )
    labels_layer = base.mark_text(dy=-8, fontSize=11, color=palette.INK_SECONDARY).encode(
        text=alt.Text("positive_rate:Q", format=".0%")
    )
    return _styled(alt.layer(bars, labels_layer))


def prevalence_line(frame, column, target, bands=6):
    cut = pd.qcut(frame[column], bands, duplicates="drop")
    grouped = frame.groupby(cut, observed=True).agg(
        centre=(column, "mean"), positive_rate=(target, "mean"), patients=(target, "size")
    )
    data = grouped.reset_index(drop=True)
    base = alt.Chart(data).encode(
        x=alt.X("centre:Q", title=f"{column}, band centre", scale=alt.Scale(nice=False), axis=alt.Axis(grid=False)),
        y=alt.Y("positive_rate:Q", title="diabetic share", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5)),
        tooltip=[
            alt.Tooltip("centre:Q", title="band centre", format=".1f"),
            alt.Tooltip("positive_rate:Q", title="diabetic share", format=".1%"),
            alt.Tooltip("patients:Q", title="patients", format=","),
        ],
    )
    band = base.mark_area(color=palette.ACCENT, opacity=0.12)
    line = base.mark_line(color=palette.ACCENT, strokeWidth=2)
    dots = base.mark_point(filled=True, size=70, color=palette.ACCENT, stroke=palette.SURFACE, strokeWidth=2)
    overall = (
        alt.Chart(pd.DataFrame({"overall": [frame[target].mean()]}))
        .mark_rule(color=palette.INK, strokeWidth=1.5)
        .encode(y="overall:Q", tooltip=alt.Tooltip("overall:Q", title="cohort share", format=".1%"))
    )
    return _styled(alt.layer(band, line, dots, overall))

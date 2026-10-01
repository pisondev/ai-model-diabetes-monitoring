import numpy as np
import pandas as pd

from .config import (
    DERIVED,
    DUMMY_ROWS,
    DUMMY_SEED,
    PROCESSED_DIR,
    dataset_config,
    numeric_features,
    schema,
)


def processed_path():
    return PROCESSED_DIR / dataset_config()["milestone_2"]["processed_file"]


def engineered_path():
    """The frame carrying the derived clinical bands, declared when the contract names it."""
    declared = dataset_config()["milestone_2"].get("engineered_file")
    if declared:
        return PROCESSED_DIR / declared
    base = processed_path()
    return base.with_name(f"{base.stem}_engineered{base.suffix}")


def load_dataset():
    """Returns the richest frame available plus which of the three sources it came from."""
    for path, source in ((engineered_path(), "engineered"), (processed_path(), "processed")):
        if path.exists():
            return pd.read_csv(path), source
    return make_dummy_frame(), "dummy"


def derived_column(frame, name):
    column = DERIVED[name]
    return column if column in frame.columns else None


def make_dummy_frame(rows=DUMMY_ROWS, seed=DUMMY_SEED):
    generator = np.random.default_rng(seed)
    spec = schema()
    columns = {name: _draw(generator, field, rows) for name, field in spec["features"].items()}
    target = spec["target"]
    columns[target["name"]] = generator.binomial(1, target["positive_rate"], rows)
    return pd.DataFrame(columns)


def _draw(generator, field, rows):
    kind = field["type"]
    if kind == "categorical":
        return generator.choice(field["values"], rows)
    if kind == "binary":
        return generator.binomial(1, field["positive_rate"], rows)
    values = generator.uniform(field["min"], field["max"], rows)
    if kind == "int":
        return values.astype(int)
    return values.round(field.get("decimals", 2))


def implausible_zeros(frame):
    """Zero counts in fields whose declared minimum rules zero out, the dataset's hidden missingness."""
    fields = schema()["features"]
    ruled_out = [
        name
        for name, field in fields.items()
        if field["type"] in {"int", "float"} and field["min"] > 0 and name in frame.columns
    ]
    return {name: int((frame[name] == 0).sum()) for name in ruled_out}


def top_risk_patients(frame, target, limit=6):
    """The cohort's sharpest cases, ranked by the derived score when it is present."""
    ranking = [column for column in (DERIVED["risk_score"], "Glucose", "Insulin") if column in frame.columns]
    if not ranking:
        ranking = [target]
    ordered = frame.sort_values(ranking, ascending=False).head(limit)
    return ordered.assign(patient=[f"#P-{index:03d}" for index in ordered.index])


def imputation_spikes(frame, target, min_share=0.05, min_gap=0.25):
    """Exact values repeated so often, and so tied to one diagnosis, that they look imputed.

    A fill value computed per outcome class leaves a pile of identical readings whose
    diagnosis share sits far from the cohort's. That is label information inside a feature,
    so any model trained on the column learns it.
    """
    cohort_rate = frame[target].mean()
    spikes = []
    for column in numeric_features():
        if column not in frame.columns:
            continue
        for value, count in frame[column].value_counts().items():
            share = count / len(frame)
            rate = frame.loc[frame[column] == value, target].mean()
            if share >= min_share and abs(rate - cohort_rate) >= min_gap:
                spikes.append({"field": column, "value": float(value), "patients": int(count),
                               "share": float(share), "positive_rate": float(rate)})
    return sorted(spikes, key=lambda row: -row["patients"])

import numpy as np
import pandas as pd

from .config import DUMMY_ROWS, DUMMY_SEED, PROCESSED_DIR, dataset_config, schema


def processed_path():
    return PROCESSED_DIR / dataset_config()["milestone_2"]["processed_file"]


def load_dataset():
    """Returns the frame plus the source it came from, either processed or dummy."""
    path = processed_path()
    if path.exists():
        return pd.read_csv(path), "processed"
    return make_dummy_frame(), "dummy"


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

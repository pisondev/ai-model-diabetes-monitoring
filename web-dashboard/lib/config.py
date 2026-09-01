from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data-engineering"
PROCESSED_DIR = DATA_ROOT / "processed"

APP_TITLE = "Diabetes Risk Screening"
APP_SUBTITLE = "Hybrid classification for healthcare risk screening"
DISCLAIMER = (
    "Academic prototype for the AI Model Engineering course. Output is not a medical "
    "diagnosis and must not be used to make clinical decisions."
)
DUMMY_ROWS = 500
DUMMY_SEED = 42


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def schema():
    return load_yaml(DATA_ROOT / "schema.yaml")


def dataset_config():
    return load_yaml(DATA_ROOT / "dataset.yaml")


def feature_names():
    return list(schema()["features"])


def target_name():
    return schema()["target"]["name"]


def features_of_type(*kinds):
    return [name for name, field in schema()["features"].items() if field["type"] in kinds]


def numeric_features():
    return features_of_type("int", "float")


def categorical_features():
    return features_of_type("categorical")

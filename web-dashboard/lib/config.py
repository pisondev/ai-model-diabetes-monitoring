from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data-engineering"
PROCESSED_DIR = DATA_ROOT / "processed"
MODELS_DIR = REPO_ROOT / "modeling" / "models"
MODEL_FILE = MODELS_DIR / "screening_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"

APP_TITLE = "Diabetes Risk Screening"
APP_SUBTITLE = "Screening overview for the frozen Pima cohort"
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

# clinical roles the screens lead with, named here so a contract rename degrades to a
# stand-in field instead of a KeyError
ROLES = {
    "primary": "Glucose",
    "secondary": "Insulin",
    "body": "BMI",
    "age": "Age",
}

# columns the cleaning pipeline derives, absent whenever only the contract file is loaded
DERIVED = {
    "risk_score": "Metabolic_Risk_Score",
    "glucose_band": "Glucose_Risk",
    "body_band": "BMI_Category",
    "age_band": "Age_Group",
}


def role(name):
    """The contract field that plays a clinical role, by name when it exists and by position otherwise."""
    names = numeric_features()
    wanted = ROLES[name]
    if wanted in names:
        return wanted
    return names[min(list(ROLES).index(name), len(names) - 1)]

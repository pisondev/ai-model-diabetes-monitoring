"""Reads the cohort the way a model must see it: raw, with the impossible zeros as missing.

The frozen file in data-engineering/processed refills a missing reading with the median of
that patient's own Outcome class, so its features carry the label. Training reads the raw
download instead and imputes inside the pipeline, fitted on training folds only.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data-engineering"
# a zero in any of these is physiologically impossible, so it is a missing reading in disguise
IMPOSSIBLE_ZERO = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
SOURCE_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"


def contract():
    with (DATA_ROOT / "dataset.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def schema():
    with (DATA_ROOT / "schema.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def feature_names():
    return list(schema()["features"])


def target_name():
    return schema()["target"]["name"]


def raw_path():
    return REPO_ROOT / contract()["acquisition"]["raw_path"]


def fetch_raw(force=False):
    """Downloads the raw file when it is absent and checks it against the recorded sha256."""
    path = raw_path()
    expected = contract()["acquisition"]["raw_sha256"]
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit("raw file sha256 is %s, the contract records %s" % (digest, expected))
    return path


def load():
    """Returns X and y from the raw file, with impossible zeros turned into NaN."""
    frame = pd.read_csv(fetch_raw())
    target = target_name()
    columns = [name for name in feature_names() if name in frame.columns]
    features = frame[columns].copy()
    for column in IMPOSSIBLE_ZERO:
        if column in features.columns:
            features[column] = features[column].replace(0, np.nan)
    return features, frame[target]

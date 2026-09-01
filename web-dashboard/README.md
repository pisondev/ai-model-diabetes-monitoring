# Web dashboard

Owner: **Pison**. Streamlit application. Nobody else edits inside this folder.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS and Linux
pip install -r requirements.txt
streamlit run Home.py
pytest
```

Run both commands from inside `web-dashboard/`.

## Layout

| Path | Contents |
|---|---|
| `Home.py` | Entry page, project status, dataset of record |
| `pages/` | One file per screen, Streamlit orders them by the numeric prefix |
| `lib/config.py` | Paths, application constants, contract readers |
| `lib/data.py` | `load_dataset()` and the dummy generator |
| `lib/predictor.py` | `Prediction`, `DummyPredictor`, `load_predictor()` |
| `lib/ui.py` | Page header, data source badge, disclaimer |
| `tests/` | Contract tests: dummy data must obey `schema.yaml` |

## How it stays honest without real data

`load_dataset()` returns `data-engineering/processed/dataset_m2_v1.csv` when that file
exists and generated dummy rows when it does not, and every page prints which of the two it
got. The dummy rows are built from `data-engineering/schema.yaml`, so the fake data always
has the real column names, types and ranges. There is no separate fixture to keep in sync
and no code change on the day the real dataset lands.

The screening page works the same way. `DummyPredictor` is deterministic and labelled as a
placeholder on screen; when the modeling side publishes an artifact, only `load_predictor()`
changes.

## Pages and what fills them later

| Page | Reads now | Reads later |
|---|---|---|
| Dataset Overview | Dummy frame, field contract | Frozen dataset |
| Data Quality | Duplicate and missing counts, class balance of the dummy frame | The same metrics on the real dataset |
| Risk Screening | Form generated from the schema, placeholder score | Trained model |
| Model Performance | Static pending table | Metrics published by the modeling side |

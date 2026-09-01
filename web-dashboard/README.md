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
| `.streamlit/config.toml` | Pinned light theme and the palette every page reads |
| `Home.py` | Entry page, project status, dataset of record |
| `pages/` | One file per screen, Streamlit orders them by the numeric prefix |
| `lib/config.py` | Paths, application constants, contract readers |
| `lib/data.py` | `load_dataset()` and the dummy generator |
| `lib/predictor.py` | `Prediction`, `DummyPredictor`, `load_predictor()` |
| `lib/ui.py` | Page header, light mode rules, data source badge, disclaimer |
| `tests/` | Contract tests: dummy data must obey `schema.yaml`, theme must stay light |

## Theme

The app is pinned to light. `.streamlit/config.toml` holds the palette, so the screens no
longer follow the operating system and a machine set to dark renders exactly what a machine
set to light renders. Streamlit leaves the CSS `color-scheme` unset, which lets the browser
carry on painting scrollbars and native controls dark on an otherwise light page, so
`page()` injects the two rules that close that gap and reads their colours from the same
config rather than keeping a second copy.

`tests/test_theme.py` guards both halves: the config has to stay light, and every screen has
to go through `page()`, which is what puts the rules on the page.

One case the server cannot reach. A browser where someone picked Dark by hand keeps that
choice in local storage and it wins over the config. Undo it in the three dot menu, under
Settings, Appearance.

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

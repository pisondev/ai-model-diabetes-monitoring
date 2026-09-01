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
| `lib/palette.py` | Chart colours, one validated set for the whole application |
| `lib/charts.py` | Every chart, one function each, pandas in and Altair out |
| `lib/ui.py` | Page header, light mode rules, data source badge, disclaimer |
| `tests/` | Contract tests: dummy data obeys `schema.yaml`, theme stays light, charts keep their form |

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

## Charts

Altair, which already ships inside Streamlit and is listed in `requirements.txt` anyway
because the code imports it directly. Every chart is a function in `lib/charts.py` that
takes a frame and returns a finished chart, which is what makes them testable without a
browser.

**The form follows the job.** Magnitude is a bar, part-to-whole is a donut or a stacked
bar, trend is a line, spread is a boxplot, polarity is a diverging heatmap, and a single
ratio against a limit is a meter. A headline number is a stat tile, not a one-bar chart.

**Aggregation happens in pandas, never in the browser.** A histogram sends its bins, not
its rows; a boxplot sends five numbers per field; the cumulative curve sends 200
quantiles. Only the scatter carries record-level rows and it samples down to a fixed cap
with a fixed seed. That keeps the page the same size on 500 dummy rows and on the real
100k, and `tests/test_charts.py` asserts it.

**Colour is assigned by the job it does, not by taste.** Two label classes keep the same
two hues on every page they appear on, ordered age bands use a single-hue ramp, the
correlation matrix uses two poles around a neutral middle, and a folded tail is the only
thing that gets grey. The set is fixed in `lib/palette.py` and was checked for
colour-vision separation and contrast against the white surface the app actually renders
on.

| Page | Charts |
|---|---|
| Home | Class balance bar, gender donut, smoking history bar with the No Info level highlighted |
| Dataset Overview | Age and BMI histograms, HbA1c against blood glucose scattered by label, positive rate by age band |
| Data Quality | Completeness bar, boxplot small multiples, correlation heatmap, positive rate by gender and by smoking history |
| Risk Screening | Score meter against the threshold, cumulative HbA1c curve marking the entered record, dumbbell of the record against the cohort median |
| Model Performance | The always-negative rule scored on the loaded dataset, accuracy against prevalence |

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
| Data Quality | Duplicate and missing counts, spread and class balance of the dummy frame | The same charts on the real dataset |
| Risk Screening | Form generated from the schema, placeholder score | Trained model |
| Model Performance | The majority-class floor computed from the loaded dataset | Metrics published by the modeling side |

The Model Performance page has no fabricated numbers on it. Until an experiment is
published it scores the always-negative rule against whichever dataset is loaded, which is
a real result and the floor a trained model has to clear.

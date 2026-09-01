# Data engineering

Owner: **Axelle**. Everything about the dataset lives here, from the raw download to the
Milestone 2 reports. Nobody else edits inside this folder.

## Layout

| Path | Contents | Committed |
|---|---|---|
| `raw/` | The untouched download, exactly as acquired | No |
| `interim/` | Intermediate steps of the cleaning pipeline | No |
| `processed/` | Model-ready output | Only `dataset_m2_*.csv` |
| `notebooks/` | Analysis notebooks | Yes, outputs stripped |
| `reports/` | Milestone 2 deliverables | Yes |
| `schema.yaml` | Feature contract, read by the dashboard | Yes |
| `dataset.yaml` | Which dataset is active, provenance, frozen version | Yes |

**The raw file is immutable.** Nothing writes back into `raw/`. Every transformation reads
from `raw/` and writes to `interim/` or `processed/`, which is what makes the before and
after evidence in the quality report verifiable.

## Setup

```
pip install -r requirements.txt
jupyter lab
```

## Milestone 2 checklist

| Item | Where | Note |
|---|---|---|
| Lab notebook | `notebooks/02_data_engineering_lab.ipynb` | Filename fixed by the lecturer, must run top to bottom on a fresh kernel |
| Acquisition log | `reports/acquisition-log.md` | At least five concrete entries: source, owner, date, scope, licence, raw location |
| Cleaning log | `reports/cleaning-log.md` | At least five entries, every changed value traceable to a rule |
| Dataset Report | `reports/dataset-report.md` | Source, scope, schema summary, version, limitations |
| Data Dictionary | `reports/data-dictionary.md` | Must agree with `schema.yaml` field for field |
| Data Quality Report | `reports/data-quality-report.md` | Missingness, duplicates, invalid values, outliers, actions, quantified |
| Data Profiling Report | `reports/data-profiling-report.md` | Distributions, class balance, correlations, three findings that change the modeling plan |
| Frozen dataset | `processed/dataset_m2_v1.csv` | Name it in `dataset.yaml` when it exists |
| Concept answers | `reports/concept-answers.md` | Two chapter questions, 100 to 150 words each |

Required figures: a before and after table of missing values for at least five features,
one boxplot, one class-count chart. Each figure carries one sentence saying what it means
and what was decided because of it.

## Two things the rest of the project depends on

1. **`schema.yaml`** is what the dashboard builds its input form and its dummy data from.
   Add a field there and it appears in the UI with no UI change. Change it in a way that
   removes or renames a field and you break the dashboard, so that PR is labelled
   `contract` and Pison reviews it.
2. **`processed/dataset_m2_v1.csv`** is what the dashboard switches to automatically. Until
   that file exists the dashboard serves dummy rows and says so on every page. Nothing has
   to be wired up when it lands; drop the file in and the placeholder disappears.

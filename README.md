# Hybrid Diabetes Classification for Healthcare Risk Screening

Group A project for **AI Model Engineering**, S1 Ilmu Komputer, Universitas Gadjah Mada.

| Role | Member | Works in |
|---|---|---|
| Data Engineer | Axelle Chandra (24/533796/PA/22614) | `data-engineering/` |
| Software Engineer | Pison Golda Mountera (24/543770/PA/23107) | `web-dashboard/` |
| AI Engineer | Irfan Fadilah Rafif (22/497042/PA/21389) | `modeling/`, added when the baselines start |

The system classifies diabetes status from medical and demographic records, built as an
engineering pipeline rather than a single notebook: acquisition and cleaning, a frozen
dataset version, baseline models, a stacking ensemble, and a dashboard that serves the
result through one consistent preprocessing path.

## Layout

One folder per person, so two people are never in the same file.

```
data-engineering/   dataset, cleaning, profiling, Milestone 2 reports    Axelle
web-dashboard/      Streamlit application                                Pison
```

Each folder has its own README and its own `requirements.txt`. Start there.

## Current state

Milestone 1 (planning) is submitted. Milestone 2 is in progress on the data side. No model
exists yet, so the dashboard runs on generated dummy data and a placeholder predictor, and
says so on every page.

## The seam between the two folders

Two files, both owned by the data side, are all that connect them:

| File | Meaning | Effect on the dashboard |
|---|---|---|
| `data-engineering/schema.yaml` | Field names, types, valid values, target | The input form and the dummy data are generated from it |
| `data-engineering/processed/dataset_m2_v1.csv` | The frozen milestone dataset | The dashboard switches to it automatically once the file exists |

Nothing has to be wired up when the real dataset arrives, and neither person waits for the
other to start. `CONTRIBUTING.md` covers branches, ownership and the rule for changing a
contract.

## Data policy

The raw download is not committed. The one exception is the frozen milestone dataset, which
the assignment asks to be linked from the repository, so `dataset_m2_*.csv` is allowed
explicitly in `.gitignore` while everything else under `data-engineering/` stays untracked.
Anyone can rebuild `raw/` from the source and checksum recorded in `dataset.yaml`.

## Limitations

This is coursework. The output is not a medical diagnosis and is not validated on any real
population. HbA1c and blood glucose are close to definitional for the label, so a high score
on this dataset says more about that relationship than about the model's ability to
generalise, and the evaluation has to say so.

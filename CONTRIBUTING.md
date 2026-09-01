# Working in this repository

Three people, one repository, five weeks. The rules below exist so that two members never
have to edit the same lines, and nobody is blocked waiting for someone else's work.

## 1. One folder per person

| Folder | Owner | Contains |
|---|---|---|
| `data-engineering/` | Axelle | Dataset, cleaning, profiling, notebooks, Milestone 2 reports |
| `web-dashboard/` | Pison | Streamlit application and its tests |
| `modeling/` | Irfan | Baselines, stacking ensemble, model artifact. Created when that work starts |
| Root files | Pison | `README.md`, `CONTRIBUTING.md`, `.gitignore`, `.github/` |

Nobody edits inside a folder they do not own. If you need a change there, open a pull
request and tag the owner. Each folder keeps its own `requirements.txt`, so the file that
three people would otherwise edit on the same day no longer exists.

## 2. The contract between folders

Two files connect the data side to the dashboard, and both belong to Axelle:

1. **`data-engineering/schema.yaml`** lists fields, types, valid values and the target. The
   dashboard generates its input form and its dummy data from this file, and the data
   dictionary in `reports/` is the human readable copy of the same thing.
2. **`data-engineering/processed/dataset_m2_v1.csv`**, named in `dataset.yaml`, is the
   frozen dataset. The dashboard loads it as soon as it exists and falls back to dummy rows
   until then.

When the modeling folder arrives it publishes a third contract, `predict(record)`, which the
dashboard already calls through `lib/predictor.py`.

**Changing a contract** breaks other people's work in a way git cannot detect. Removing or
renaming a field in `schema.yaml`, or changing the frozen dataset name, goes in a pull
request labelled `contract`, reviewed by whoever consumes it, with one line saying what
moved. Adding a field is safe and needs no ceremony.

## 3. Branches

`main` is always runnable. Nobody pushes to it directly.

```
de/<topic>    data work         example: de/cleaning-missing-values
ai/<topic>    modeling work     example: ai/baseline-logreg
se/<topic>    dashboard work    example: se/dummy-data-shell
```

Branch from the latest `main`, keep it short lived, open a pull request, squash merge. A
branch that lives longer than a few days will conflict with something; split the work.

On GitHub: require a pull request before merging and one approval.

## 4. Notebooks

A notebook is a JSON file, so two people editing one produces a conflict git cannot resolve,
and committed output cells make the diff unreadable.

- One owner per notebook. Never edit someone else's.
- Run `nbstripout --install` once after cloning, then outputs are stripped on commit.
- A notebook must run top to bottom on a fresh kernel, which is the lecturer's acceptance
  condition for the Chapter 2 lab.

## 5. Commits

English, imperative mood, Conventional Commits, one logical change each.

```
feat: add stratified split to the cleaning pipeline
fix: correct bmi range in the feature contract
docs: write the acquisition log
exp: record baseline logistic regression metrics
```

Commit history is part of what the lecturer can inspect. Keep it incremental, and leave the
repository runnable at every commit.

## 6. Before opening a pull request

```
pytest                        # from web-dashboard/, if the dashboard changed
streamlit run Home.py         # from web-dashboard/, if a page changed
```

The checklist in `.github/pull_request_template.md` is short on purpose: stay inside your
folder, do not commit data or model artifacts, strip notebook outputs.

## 7. Local setup

```
git clone https://github.com/pisondev/ai-model-diabetes-monitoring.git
cd ai-model-diabetes-monitoring
git config user.name "<your name>"
git config user.email "<your email>"
python -m venv .venv
.venv\Scripts\activate
pip install -r web-dashboard/requirements.txt        # dashboard work
pip install -r data-engineering/requirements.txt     # data work
```

# Modeling

Trains the screening model the dashboard serves. One command, fixed seeds, and the metrics
written next to the artifact.

```
pip install -r modeling/requirements.txt
python modeling/train.py
```

The run takes about a minute and writes two files into `modeling/models/`, both ignored by
git because they are build output, not source:

| File | Contents |
|---|---|
| `screening_model.joblib` | The fitted pipeline, the decision threshold, and the feature order |
| `metrics.json` | Metrics, leaderboard, seed, split, environment, timestamp |

The dashboard picks both up on its next run. With no artifact present it falls back to the
placeholder predictor and says so on screen.

## It does not train on the frozen dataset

`data-engineering/processed/dataset_m2_v1.csv` fills each missing reading with the median of
that patient's own `Outcome` class, so its features carry the label. Training reads the raw
download instead, turns the impossible zeros back into missing values, and imputes inside the
pipeline, where the imputer only ever sees a training fold. `src/dataset.py` downloads the raw
file on first use and refuses to continue unless its sha256 matches the one recorded in
`data-engineering/dataset.yaml`.

Measured on this cohort, training on the frozen file instead would have inflated Random Forest
PR-AUC from 0.722 to 0.913 and recall from 0.634 to 0.817. None of that gain is real.

## What it builds

Four level-0 learners, each a `Pipeline` so preprocessing fits inside whatever split surrounds
it: logistic regression, random forest, XGBoost, and a calibrated SVM. Their out-of-fold
probabilities feed a logistic regression meta-learner, which is kept deliberately simple so it
cannot memorise the base learners on 614 training rows.

The threshold is not left at 0.5. It is picked on out-of-fold predictions as the best F1 among
thresholds that still hold recall at 0.70 or above, because a missed patient costs more than a
false alarm in screening.

## The run that is live now

Stratified 80 to 20 split, seed 42, stratified 10-fold cross validation on the training part.

| Candidate | CV PR-AUC |
|---|---|
| stacking | 0.727 |
| logistic regression | 0.725 |
| random forest | 0.704 |
| xgboost | 0.682 |
| svm | 0.659 |

On the held-out 154 patients, at threshold 0.366: PR-AUC 0.674, recall 0.852, precision 0.568,
F1 0.681, ROC-AUC 0.820. The rule that always answers negative scores PR-AUC 0.349 and recall 0
on the same data. Of 54 diabetic patients the model finds 46 and misses 8, at the cost of 35
false alarms.

The ensemble leads the table by 0.002 over plain logistic regression. That is thin, and it is
reported as thin. Rerun `train.py` to reproduce every number here.

## Layout

| Path | Contents |
|---|---|
| `src/dataset.py` | Fetches and verifies the raw file, turns impossible zeros into missing |
| `src/models.py` | The four candidates and the stacking ensemble, all as pipelines |
| `train.py` | Runs the comparison, picks the threshold, writes the artifact and metrics |
| `tests/` | Contract tests: zeros become missing, every candidate carries its own imputer, the threshold search respects the recall floor |

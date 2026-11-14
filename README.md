# Big Data Assignment 2: Predicting GitHub User Activity

For each test user, predict whether they will perform each of 6 activities
(`open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push`)
during 2026-11-22 00:00 – 11-29 00:00 (Taiwan time), i.e. 2026-11-21 16:00 – 11-28 16:00 UTC.

## Data Source

The data comes from [GH Archive](https://www.gharchive.org/), a public archive of GitHub events.
The 8 tables are built from the GH Archive hour files with the course preprocessing script
(`assignment2/scripts/preprocess.py`, unmodified), which keeps the 9,000 cohort users and
anonymizes ids, names and text.

Dataset time period used by the model: **2025-12-01 00:00 UTC – END** (all GH Archive hour files
in this range; END is given to `scripts/build_data.sh`, see below).

## Layout

```
.
├── src/
│   ├── data.py           # loading, point-in-time snapshot, labels, data-gap detection, scoring
│   ├── features.py       # point-in-time user features (activity, relational, data coverage)
│   ├── train_gbdt.py     # LightGBM baseline: weekly samples, time-based validation, thresholds
│   ├── predict.py        # GBDT: retrain on all weeks, write the submission csv, check its format
│   ├── rdl_data.py       # the 8 tables as a RelBench Database (PK/FK edges, row times)
│   ├── train_rdl.py      # Relational Deep Learning: temporal heterogeneous GNN
│   └── ensemble.py       # GBDT + RDL probability ensemble, thresholds, submission csv
├── scripts/
│   ├── download_gharchive.sh  # download GH Archive hour files
│   ├── build_data.sh          # download + course preprocessing -> data/db/*.parquet
│   └── run_final.sh           # full pipeline: GBDT + RDL (3 seeds) + ensemble
├── environment.yml       # conda environment (exact versions)
├── requirements.txt      # pip packages besides torch / PyG (used by the Dockerfile)
├── Dockerfile
├── assignment2/          # course repository: preprocess.py, evaluate.py, test_input.csv (not in git)
├── gharchive/            # GH Archive hour files (not in git, ~215 GB for 2025-12-01 .. 2026-11-08)
└── data/db/              # the 8 tables written by preprocess.py (not in git)
```

## Setup

```bash
conda env create -f environment.yml
conda activate bigdata-a2
```

Or with Docker (data is mounted, not copied into the image):

```bash
docker build -t bigdata-a2 .
docker run --gpus all -it --rm -v "$PWD":/work bigdata-a2 bash scripts/run_final.sh
```

## Data

```bash
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
END=2026-11-21T12 bash scripts/build_data.sh
```

`build_data.sh` downloads every GH Archive hour file from 2025-12-01 00:00 UTC to `END` (UTC,
inclusive) into `gharchive/` and runs the course script
`python assignment2/scripts/preprocess.py --raw gharchive --out data`.
To use another data folder, set `A2_DB=<folder>/db`.

## Run

```bash
bash scripts/run_final.sh outputs/output_<studentID>.csv
```

or step by step:

```bash
python src/train_gbdt.py --tag base            # GBDT -> outputs/gbdt/base/valid_report.json
python src/predict.py --tag base --out outputs/gbdt/base/output.csv
python src/train_rdl.py --tag l3_s42 --seed 42 --layers 3 --epochs 15   # RDL validation run
python src/train_rdl.py --tag l3_final_s42 --seed 42 --layers 3 --epochs 15 --final --stop-at <epoch>
python src/ensemble.py --gbdt base --rdl-valid l3_s42 --rdl-test l3_final_s42 --out outputs/output.csv
```

Figures and numbers of the report (after run_final.sh; Graphviz `dot` renders the .gv files):

```bash
python src/analysis.py --gbdt base     # outputs/analysis.json
python src/q1_schema.py                # Q1 schema graph + data period
python src/q3_data_gaps.py             # Q3 collection gaps
python src/q3_drift.py                 # Q3 weekly positive rate
python src/q4_framework.py             # Q4 framework figure
```

Each script that writes a submission also runs the official format check
(`assignment2/scripts/evaluate.py --input assignment2/submission/test_input.csv`).

Training samples are weekly: the label window starts on Saturdays 16:00 UTC like the test week,
features use data up to 4 hours earlier (like the final data, up to 11-21 12:00 UTC), and users
are those with at least 5 activity records in the 90 days before the cutoff. The last 4 weeks
are the validation set. Random seeds are fixed (42, 43, 44).

## Notes on the data

- Labels are built from the tables by the activity time of each record
  (`PullRequests.created_at`, non-PR `Issues.created_at`, `IssueComments.created_at`,
  `PRReviews.submitted_at`, `PRReviewComments.created_at`, `Pushes.created_at`).
- Features for a sample at time t only use rows whose row-time column is earlier than t,
  and known-later columns (`merged_at`, `merged_by_id`, `closed_at`, `state_reason`,
  `dismissed_at`) only when their value is earlier than t (`data.snapshot`).
- Users are matched by `github_user_id` (an anonymized code); row ids depend on the data period.
- Text columns are masked by the preprocessing script; only their lengths are used.
- GH Archive itself has gaps (hours with missing or incomplete events). Features include the
  data coverage of each activity before the cutoff (`cov_*`, `lag_*_h`, `*_adj`).
- `pytorch-frame` 0.2.5 is needed for `train_rdl.py`; `train_rdl.py` patches
  `relbench.modeling.graph.to_unix_time`, which fails under pandas 3 (read-only arrays).

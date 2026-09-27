# Portable credit default experiment (v2)

Package contains pipeline code, runtime modules, dependencies, and local UCI/Lending Club data. It excludes historical output archives and manuscript files. South German Credit downloads from UCI when requested and no local copy exists; internet access is required for that first download.

## Setup

Use Python 3.11 or 3.12. From extracted folder:

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn scipy
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. On macOS, install OpenMP if LightGBM reports a missing OpenMP library.

## Full runs

Run each command from this extracted folder. All commands include calibration, robustness, and paired-bootstrap analyses. UCI and South German also have supported demographic/group fields for fairness curves and validation-fitted fairness post-processing. The packaged Lending Club sample has no group fields, so those fairness outputs are skipped for that dataset.

UCI:

```bash
python pipeline.py \
  --dataset uci \
  --seeds 42 99 123 326 456 515 689 777 872 999 \
  --tune-iter 12 \
  --revision-analyses \
  --revision-analysis-n 100 \
  --bootstrap-reps 2000
```

South German Credit:

```bash
python pipeline.py \
  --dataset south_german \
  --seeds 42 99 123 326 456 515 689 777 872 999 \
  --tune-iter 12 \
  --revision-analyses \
  --revision-analysis-n 100 \
  --bootstrap-reps 2000
```

Temporal Lending Club:

```bash
python pipeline.py \
  --dataset lending_club \
  --split-mode temporal \
  --temporal-column issue_time \
  --seeds 42 99 123 326 456 515 689 777 872 999 \
  --tune-iter 12 \
  --revision-analyses \
  --revision-analysis-n 100 \
  --bootstrap-reps 2000
```

## Inputs and outputs

UCI data are bundled at `data/uci_credit.csv`; Lending Club sample is `data/lending_club_subsample_50k_by_issue_date.csv`. South German Credit downloads automatically if absent; alternatively place `south_german_credit.csv` or `SouthGermanCredit.asc` in `data/`.

Outputs are separated by dataset under `results/outputs/<dataset>/runs/` (`uci`, `south_german`, `lending_club`). Per-seed diagnostics live in `runs/seed_<seed>/results/`; aggregate files live in `runs/summary/`. Rerunning the same dataset and seed overwrites matching files in its folder.

The Lending Club split is row-chronological; loans sharing issue months can cross the split boundary, so it is not a strict calendar-month holdout.

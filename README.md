# Credit default prediction beyond AUROC

An empirical benchmark of six classifier families: logistic regression, Random
Forest, XGBoost, LightGBM, CatBoost, and TabNet. The project compares discrimination,
imbalance handling, decision thresholds, probability calibration, synthetic
perturbation sensitivity, explanation diagnostics, and group disparities.

The contribution is an empirical comparison using established methods. It is not
a new learning algorithm, a validated trustworthiness measure, or evidence of
deployment readiness. Findings apply to the evaluated portfolios and protocols.

## Current result snapshot

The active tree, `results/outputs/`, contains **32 runs across three datasets**.
Each run reports all six classifier families.

| Dataset | Labelled rows | Evaluation | Runs | Fairness diagnostic groups |
|---|---:|---|---:|---|
| `uci` | 30,000 | Stratified random, 80/20 | 10 seeds | SEX, EDUCATION, MARRIAGE |
| `south_german` | 1,000 | Stratified random, 80/20 | 10 seeds | personal_status_sex, foreign_worker, age_group (<=25 / >25) |
| `prosper_temporal` | 72,067 | Six rolling-origin cutoffs; 12-month default horizon; 6-month test windows | 6 cutoffs x 2 seeds | None available |

Random-split seeds: `42 99 123 326 456 515 689 777 872 999`. Prosper uses seeds
`42 99` at cutoffs `2010-01 2010-07 2011-01 2011-07 2012-01 2012-07`.
Prosper's row count is the complete labelled dataset, not one split's size.

The loader also supports the packaged 50,000-row Lending Club sample. **No Lending
Club results are present in the active snapshot.** The sample lacks the fields
required by this implementation for a censoring-aware temporal split and supports
random splitting only. The current results are not a four-dataset experiment or
cross-platform external validation of a fixed fitted model.

### Main observations

| Dataset | Highest mean AUROC | Lowest mean Brier score |
|---|---|---|
| UCI | CatBoost and LightGBM: 0.7879 | LightGBM: 0.1324 |
| South German | Random Forest: 0.7902 | Random Forest: 0.1662 |
| Prosper | CatBoost: 0.6822 | LightGBM: 0.0618 |

These are descriptive means, not declarations of statistically distinct winners.
Full summaries: [UCI](results/outputs/uci/runs/summary/all_run_metric_summary.csv),
[South German](results/outputs/south_german/runs/summary/all_run_metric_summary.csv),
and [Prosper](results/outputs/prosper_temporal/runs/summary/all_run_metric_summary.csv).

- Similar discrimination can conceal different probability quality. On Prosper,
  only LightGBM has positive held-out Brier skill in all 12 runs relative to a
  constant predictor using each test partition's prevalence.
- OOF-selected thresholds improve mean F1 over a fixed 0.5 threshold by 0.0267,
  0.0376, and 0.1434 on UCI, South German, and Prosper, respectively. These are
  equally weighted paired differences across 60, 60, and 72 champion-family
  strategy/run comparisons. Threshold changes leave AUROC and Brier unchanged.
- Synthetic oversampling does not provide a consistent advantage. UCI's
  OOF-selected champions use `None` in eight runs and `ClassWeight` in two.
- No run records a feasible `reliability_selected_model`. On Prosper the fairness
  component is unavailable; this is not evidence of a measured fairness violation.

Differences between datasets cannot be attributed solely to split design: their
populations, features, and target definitions differ.

## Selection and evaluation protocol

1. **Fold-local preprocessing and sampling.** Models use an
   `imblearn.pipeline.Pipeline(preprocessor -> sampler -> model)`. Median
   imputation, 1st/99th-percentile winsorisation, scaling, and sampling are fitted
   within each training fold. Nominal vocabularies and the missing-indicator
   column layout are fixed from the outer training partition. Synthetic rows
   never enter a scoring fold.
2. **Nominal-aware sampling.** Nominal columns are excluded from interpolation,
   continuous scaling, winsorisation, and numeric perturbations. Oversamplers
   operate on the continuous block; synthetic rows receive nominal values voted
   by their five nearest original minority neighbours. This is a custom sampler,
   not library SMOTENC. Logistic regression one-hot encodes nominal features;
   TabNet embeds them; LIME treats them as categorical.
3. **Training-only selection.** Imbalance strategy, hyperparameters, thresholds,
   champion model, calibration maps, and fairness policies use training CV/OOF
   predictions. Strategy choice is per family on default-parameter OOF AUROC;
   TabNet inherits the probe strategy. Tuning and subsequent OOF prediction reuse
   folds. **This is not fully nested CV:** OOF scores are selection diagnostics,
   not unbiased performance estimates. Early stopping uses a stratified 10% slice
   of each fit's data after preprocessing and sampling.
4. **Temporal label availability.** Prosper labels indicate default within 12
   months; censored or unresolved labels are excluded. Outer training labels must
   be observable by the cutoff. `ObservableTimeSeriesSplit` uses month-disjoint
   validation blocks and requires every inner training label to be observable
   at the validation block's start.
5. **Conditional inference.** Pairwise DeLong and McNemar tests use Holm correction
   within each run. Saved bootstrap analyses use 2,000 replicates; intervals
   condition on fitted models and selected thresholds. Seeds reuse observations.
   Prosper's two seeds share each test window, and expanding training sets overlap
   across windows. Across-run means, standard deviations, and ranks are
   descriptive, not independent-run significance tests or universal rankings.

### Protocol provenance

All 12 active Prosper manifests identify `v3_training_cv_oof_selection`; their
revision-analysis manifests identify `ObservableTimeSeriesSplit`. Saved OOF masks
match reconstruction of this splitter for all six cutoff blocks checked with
seed 42. The current snapshot contains corrected temporal-selection results;
a corrected rerun is not pending merely because older documentation says so.

UCI and South German manifests retain the historical label `v3_nested_selection`.
That label does not make their selection procedure fully nested. Describe the
actual training-only CV/OOF procedure above.

`results/outputs/reporting_corrections.json` records an earlier reporting-only
repair. Its temporal warning concerns that repair's source results, not proof
that subsequently updated Prosper runs still use unpurged CV. Consult individual
run manifests and saved predictions when interpreting provenance.

## Interpretation limits

- **Champion-only XAI.** Explanation diagnostics do not compare all six families.
  SHAP additivity checks numerical reconstruction, not causal validity or human
  usefulness. Magnitudes in different attribution output spaces are not directly
  interchangeable.
- **Restricted LIME sampling.** SHAP indices are randomly drawn and sorted. LIME
  agreement uses the first 200 selected rows; repeated-seed diagnostics use the
  first 100. These subsets concentrate near the start of test-row order rather
  than being uniform draws from the entire partition. On Prosper, the 100-row
  subsets lie within approximately the first 4.5-5.3% of test-row positions.
  Source rows are not chronologically sorted, so this is not necessarily temporal
  concentration. Treat these summaries as exploratory diagnostics of those cases.
- **Repeatability and concentration.** LIME stability uses three repeated seeds
  per case, not input perturbations or model refitting. Agreement concerns ranks
  of absolute attribution magnitudes on shared selected features, not effect
  direction. LIME retains 15 features, so concentration depends on truncation.
- **Absolute deletion/insertion metrics.** The saved `comprehensiveness` column
  is `abs(p_full - p_deleted)` and `sufficiency` is `abs(p_full - p_inserted)`,
  using a training median/mode baseline. Report these as absolute probability
  changes. A large absolute deletion change does not establish that removed
  features supported the explained class. No random-feature control is included.
- **Synthetic robustness.** Missingness replacement, block replacement, numeric
  noise, and additive shifts assess specified sensitivities. They do not establish
  resilience to adversarial attacks or actual economic shocks. Nominal features
  remain unchanged. Selection-side robustness uses an in-sample training subset
  and is descriptive, not an independent validation estimate.
- **Fairness depends on groups and objectives.** OOF-fitted thresholds need not
  meet held-out gap constraints. Some South German gaps are undefined when a
  group has no positive cases. Utility assumes FP:FN costs of 1:5. The reference
  threshold maximizes F1 while the group policy optimizes cost, so utility gains
  cannot be attributed solely to fairness intervention. Prosper lacks group data.
- **Composite scores are exploratory.** `reliability_scores.csv` uses a geometric
  mean of predictive quality, perturbation retention, explanation diagnostics,
  and inverse group gap, with chosen scaling and clipping. It is champion-only,
  not a validated trustworthiness metric or a complete six-classifier comparison.
  Report components alongside it. Missing fairness leaves the score unavailable.
- **Limited early temporal support.** The first Prosper inner training fold has
  108 loans, including seven defaults. Sensitivity to a later validation start
  has not been evaluated. Stability of selection remains uncertain.

## Outputs and reproducibility

Runs are under `results/outputs/<dataset>[_temporal]/runs/<run_id>/`, with IDs
`seed_<s>` or `cutoff_<YYYY-MM>_seed_<s>`.

| Artifact | Interpretation |
|---|---|
| `results/test_metrics.csv` | Held-out metrics; display values are rounded |
| `results/oof_metrics.csv` | Training selection diagnostics |
| `results/controlled_ablations.csv` | Champion-family strategy, threshold, and isotonic comparisons; UCI also includes feature engineering |
| `results/heldout_calibration_*.csv` | Brier skill, calibration diagnostics, and reliability bins |
| `results/robustness_perturbations.csv` | Held-out performance and decision changes under synthetic perturbations |
| `results/explanation_*.csv`, `shap_lime_agreement.csv` | Exploratory champion explanation diagnostics |
| `results/oof_fitted_fairness_postprocessing.csv` | Frozen OOF-fitted group policies and held-out comparisons |
| `results/fairness_postprocessing_bootstrap_intervals.csv` | Conditional intervals for policy differences, where groups are available |
| `results/reliability_scores.csv`, `reliability_selection.csv` | Descriptive audit score and separate constraint diagnostics |
| `results/run_manifest.json`, `revision_analysis_manifest.json`, `split_indices.json` | Settings, analysis protocol, and row membership |
| `runs/summary/all_run_*.csv`, `descriptive_model_ranks.json` | Descriptive summaries of compatible saved runs |

Prosper includes 72 row-aligned `results/predictions_*.npz` files and 72
`models/*_training.pkl` checkpoints. Prediction files contain row IDs, labels,
OOF probabilities/masks, held-out probabilities, and selected thresholds. They
permit metric reanalysis without fitting. UCI and South German retain model files
and reporting tables, but neither these prediction files nor full training
checkpoints are present for them. Raw per-instance explanation arrays are not
provided as equivalent reusable checkpoints.

Prosper manifests set `skip_ablation_grid=true`. Existing `smote_model_ablation.csv`
files may remain from earlier runs; their presence alone does not establish
corrected-protocol provenance. Use current controlled ablations and manifest-backed
summaries; exclude legacy grids unless their provenance is established.

The snapshot analysis checked all 72 Prosper prediction files against reported
AUROC, AUPRC, F1, and Brier values, with agreement within display rounding.
Dataset checksums and summary consistency were also checked. This verifies
saved-result arithmetic and selected protocol properties, not fresh training
reproduction.

Historical absolute paths in manifests refer to the producing machine. This
checkout does not include `results/outputs_original_archive/` or
`results/outputs_v2_archive/`. The CLI protects those names if archives are
supplied later.

## Publication framing

Frame the study as an empirical comparison of dataset-dependent trade-offs.
Avoid claims of a new benchmarking standard, universal superiority, complete
elimination of leakage, validated reliability, or regulatory readiness. Three
datasets broaden the earlier single-dataset evidence without establishing
generalization to all borrowers, institutions, or periods.

If further runs are unavailable, retain supported discrimination, calibration,
threshold, and ablation findings. Scope XAI, fairness, and composite analyses to
what was measured. Do not describe unperformed sensitivity checks as completed.

[Reviewer response.txt](<Reviewer response.txt>) provides a point-by-point draft,
separating available evidence, partially addressed requests, and proposed
manuscript wording. No manuscript is included here, so its edits and page/line
locations have not been verified.

## Setup and optional execution

Reading saved CSV/JSON tables requires no training environment. For execution,
`requirements.txt` pins recorded package versions; manifests record Python 3.13.5.
Check the environment against those manifests before attempting checkpoint reuse.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

This checkout contains UCI, South German, Prosper, and the Lending Club sample
in `data/`. Prosper's source export has 113,937 rows; the loader deduplicates
`LoanKey` and applies horizon labelling. Refer to originating data sources and
access terms when preparing a data-availability statement.

Full training is optional. `./run_all_full.sh` runs all three evaluated datasets
and writes to the active tree. Use another output root for a separate new run:

```bash
python pipeline.py --dataset uci --seeds 42 99 123 326 456 515 689 777 872 999 \
  --tune-iter 12 --revision-analyses --revision-analysis-n 100 --bootstrap-reps 2000 \
  --out-dir results/new_runs

python pipeline.py --dataset prosper --split-mode temporal \
  --cutoffs 2010-01 2010-07 2011-01 2011-07 2012-01 2012-07 --seeds 42 99 \
  --tune-iter 12 --revision-analyses --revision-analysis-n 100 --bootstrap-reps 2000 \
  --out-dir results/new_runs
```

Options include `--models`, `--skip-tabnet`, `--fast`, `--threshold-criterion cost`,
`--cost-fp`, `--cost-fn`, `--horizon-months`, and `--test-window-months`.
`--prosper-include-pricing` enables BorrowerAPR, BorrowerRate, and MonthlyLoanPayment;
these are excluded by default because they encode the platform's assessment.

### Checkpoint reuse and reporting repair

`--resume` reuses matching checkpoints but can still perform missing fits.
`--core-only` retains strategy/hyperparameter selection, OOF threshold selection,
and test reporting while skipping XAI and extra analysis fits. It does not reduce
the core search budget.

`--evaluate-only` requires matching full-pipeline/OOF checkpoints and never falls
back to classifier training. It excludes fitting ablation grids and external
validation fits, but may execute explainers unless `--skip-xai` is supplied.
It is not equivalent to reaggregating existing tables.

Legacy final-model reuse requires new selection to agree, dataset checksums and
row membership to match, nominal vocabularies to match, and reconstructed
predictions to reproduce saved AUROC/AUPRC/Brier within display precision.
Otherwise a training run fits the model normally. `--no-tune` uses default
parameters; it does not load saved tuned parameters.

`repair_reports.py` copies reporting artifacts to a new, separate destination
without fitting models or running explainers:

```bash
python repair_reports.py --source results/outputs --out-dir results/reporting_fixed
```

It restores the sign of rank correlations between absolute attribution magnitudes,
regenerates Table 5, labels bootstrap scope, and writes descriptive ranks. It does
not recover attribution effect directions or fix sampling, selection, or model
fits. Its historical temporal warning should not replace source-run protocol
records. Current results already contain reporting repairs. An existing
destination is never overwritten.

Incremental-rerun tests also work without pytest:

```bash
python -m unittest discover -s tests -p test_minimal_rerun.py -v
```

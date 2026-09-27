#!/bin/zsh
# Runs the three full RUN_EXPERIMENT.md dataset experiments sequentially.
# Logs: results/logs/<dataset>_full.log ; status markers appended to results/logs/status.txt
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
STATUS=results/logs/status.txt
SEEDS=(42 99 123 326 456 515 689 777 872 999)
COMMON=(--seeds $SEEDS --tune-iter 12 --revision-analyses --revision-analysis-n 100 --bootstrap-reps 2000)

run() {
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> $STATUS
  $PY pipeline.py "$@" > results/logs/${name}_full.log 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc" >> $STATUS
}

run uci          --dataset uci $COMMON
run south_german --dataset south_german $COMMON
run lending_club --dataset lending_club --split-mode temporal --temporal-column issue_time $COMMON
echo "$(date '+%F %T') ALL_DONE" >> $STATUS

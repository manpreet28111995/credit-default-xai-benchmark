#!/bin/zsh
# Protocol v3 full runs. Logs: results/logs/<name>_full.log ; markers in results/logs/status.txt
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p results/logs
STATUS=results/logs/status.txt
SEEDS=(42 99 123 326 456 515 689 777 872 999)
COMMON=(--tune-iter 12 --revision-analyses --revision-analysis-n 100 --bootstrap-reps 2000)

run() {
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> $STATUS
  $PY pipeline.py "$@" > results/logs/${name}_full.log 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc" >> $STATUS
}

# Random-split datasets: one run per seed (split + model RNG); selection uses training-only CV.
run uci          --dataset uci          --seeds $SEEDS $COMMON
run south_german --dataset south_german --seeds $SEEDS $COMMON
# Out-of-time: rolling-origin blocks are the replication unit; two seeds per block.
run prosper      --dataset prosper --split-mode temporal \
                 --cutoffs 2010-01 2010-07 2011-01 2011-07 2012-01 2012-07 \
                 --seeds 42 99 $COMMON
echo "$(date '+%F %T') ALL_DONE" >> $STATUS

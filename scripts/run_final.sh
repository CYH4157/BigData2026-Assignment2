#!/usr/bin/env bash
# Full pipeline: GBDT + RDL (3 seeds) + ensemble -> submission csv.
#   bash scripts/run_final.sh [output.csv]
# One GPU per RDL seed: GPUS="1 2 3" (default "0 1 2"). Logs go to outputs/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUTCSV=${1:-outputs/output.csv}
SEEDS=(42 43 44)
read -r -a GPU <<< "${GPUS:-0 1 2}"
RDL_ARGS="--layers 3 --epochs 15 --min-cov 0.95"
mkdir -p outputs/logs

echo "[1/4] GBDT"
$PY -u src/train_gbdt.py --tag base --min-cov 0.95 > outputs/logs/gbdt_train.log 2>&1
$PY -u src/predict.py --tag base --out outputs/gbdt/base/output.csv > outputs/logs/gbdt_predict.log 2>&1

echo "[2/4] RDL validation runs (seeds ${SEEDS[*]})"
for i in "${!SEEDS[@]}"; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=${GPU[$i]} $PY -u src/train_rdl.py --tag l3_s$s --seed $s $RDL_ARGS > outputs/logs/rdl_l3_s$s.log 2>&1 &
done
wait

EP=$($PY -c "
import json, statistics, sys
print(int(statistics.median(json.load(open(f'outputs/rdl/l3_s{s}/valid_report.json'))['epoch'] for s in sys.argv[1:])))
" "${SEEDS[@]}")
echo "[3/4] RDL final runs on all weeks, stop at epoch $EP"
for i in "${!SEEDS[@]}"; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=${GPU[$i]} $PY -u src/train_rdl.py --tag l3_final_s$s --seed $s --final --stop-at $EP $RDL_ARGS \
        > outputs/logs/rdl_l3_final_s$s.log 2>&1 &
done
wait

echo "[4/4] ensemble"
$PY -u src/ensemble.py --gbdt base \
    --rdl-valid $(printf "l3_s%s " "${SEEDS[@]}") \
    --rdl-test $(printf "l3_final_s%s " "${SEEDS[@]}") \
    --out "$OUTCSV" 2>&1 | tee outputs/logs/ensemble.log

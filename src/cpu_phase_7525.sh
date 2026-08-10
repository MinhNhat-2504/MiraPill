#!/bin/bash
# Phase 3 of the 75/25 switch: promote the 7525 caches to canonical names,
# then re-derive every downstream result (all CPU, ~minutes).
cd "/d/Project/AI Engineer/MiraPill/src" || exit 1
export PYTHONIOENCODING=utf-8

need () { [ -f "$1" ] || { echo "FATAL: missing $1"; exit 1; }; }
need fusion_out/imprints_holdout_7525.csv
need fusion_out/cure_imprints_holdout_7525.csv
need fusion_out/ogyei_imprints_7525.csv
need fusion_out/softgate_logprob_7525.csv
need fusion_out/softgate_result_7525.json
need "/d/Data/CURE_crops/cure_imprint_db_7525.csv"
need "/d/Data/ogyeiv2_crops/ogyei_imprint_db_7525.csv"

echo "== promote 7525 caches to canonical names (90/10 kept in _9010_backup) =="
cp fusion_out/imprints_holdout_7525.csv      fusion_out/imprints_holdout.csv
cp fusion_out/cure_imprints_holdout_7525.csv fusion_out/cure_imprints_holdout.csv
cp fusion_out/ogyei_imprints_7525.csv        fusion_out/ogyei_imprints.csv
cp fusion_out/softgate_logprob_7525.csv      fusion_out/softgate_logprob.csv
cp fusion_out/softgate_result_7525.json      fusion_out/softgate_result.json
cp "/d/Data/CURE_crops/cure_imprint_db.csv"      "/d/Data/CURE_crops/cure_imprint_db_9010.csv"
cp "/d/Data/CURE_crops/cure_imprint_db_7525.csv" "/d/Data/CURE_crops/cure_imprint_db.csv"
cp "/d/Data/ogyeiv2_crops/ogyei_imprint_db.csv"      "/d/Data/ogyeiv2_crops/ogyei_imprint_db_9010.csv"
cp "/d/Data/ogyeiv2_crops/ogyei_imprint_db_7525.csv" "/d/Data/ogyeiv2_crops/ogyei_imprint_db.csv"

step () {  # $1 tag, rest = command
  local tag="$1"; shift
  echo; echo "===== $tag $(date +%H:%M:%S) ====="
  "$@" 2>&1 | tee "_p3_$tag.log"
  local rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && echo "FATAL: $tag rc=$rc" && exit 1
}

step e1_fuse4fold python -u run_with_newbase.py fuse_4fold
step e2_signif    python -u run_with_newbase.py e2_significance
step e3_failure   python -u run_with_newbase.py e3_failure_analysis
step e4_openset   python -u run_with_newbase.py e4_openset
step e5_ablation  python -u run_with_newbase.py e5_ablation

step cure_fusion python -u cure_fusion.py \
  --pred_csv outputs/eval_predictions_cure_val.csv \
  --label_encoder "D:/Data/CURE_crops/folds/cure/base/label_encoder_local.pickle" \
  --db_csv "D:/Data/CURE_crops/cure_imprint_db.csv" \
  --imprint_cache fusion_out/cure_imprints_holdout.csv --lam 0.3

step ogyei_fusion python -u cure_fusion.py \
  --pred_csv outputs/eval_predictions_ogyei_val.csv \
  --label_encoder "D:/Data/ogyeiv2_crops/folds/ogyei/base/label_encoder_local.pickle" \
  --db_csv "D:/Data/ogyeiv2_crops/ogyei_imprint_db.csv" \
  --imprint_cache fusion_out/ogyei_imprints.csv --lam 0.3

step rebuttal_stats  python -u rebuttal_stats.py
step rebuttal_stats2 python -u rebuttal_stats2.py
step finalize        python -u finalize_remaining.py

echo; echo "CPU_PHASE_DONE $(date +%H:%M:%S)"

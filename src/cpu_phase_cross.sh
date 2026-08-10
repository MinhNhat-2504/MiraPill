#!/bin/bash
# Phase 3b: rebuild OGYEI 7525 db with the identified original protocol
# (max_ref_per_class=3), promote the cross-dataset caches, re-derive
# CURE/OGYEI fusion + rebuttal_stats2 + finalize_remaining.
cd "/d/Project/AI Engineer/MiraPill/src" || exit 1
export PYTHONIOENCODING=utf-8

step () {
  local tag="$1"; shift
  echo; echo "===== $tag $(date +%H:%M:%S) ====="
  "$@" 2>&1 | tee "_p3_$tag.log"
  local rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && echo "FATAL: $tag rc=$rc" && exit 1
}

step ogyei_db_cap3 python -u run_bigstack.py cure_imprint_db \
  --manifest "D:\\Data\\ogyeiv2_crops\\manifest.csv" \
  --crops_root "D:\\Data\\ogyeiv2_crops" \
  --adapter qwen_imprint_lora_7525 \
  --max_ref_per_class 3 \
  --out "D:\\Data\\ogyeiv2_crops\\ogyei_imprint_db_7525cap3.csv"

echo "== promote cross-dataset caches (originals backed up as *_9010) =="
cp "/d/Data/CURE_crops/cure_imprint_db.csv"      "/d/Data/CURE_crops/cure_imprint_db_9010.csv"
cp "/d/Data/CURE_crops/cure_imprint_db_7525.csv" "/d/Data/CURE_crops/cure_imprint_db.csv"
cp "/d/Data/ogyeiv2_crops/ogyei_imprint_db.csv"          "/d/Data/ogyeiv2_crops/ogyei_imprint_db_9010.csv"
cp "/d/Data/ogyeiv2_crops/ogyei_imprint_db_7525cap3.csv" "/d/Data/ogyeiv2_crops/ogyei_imprint_db.csv"
cp fusion_out/cure_imprints_holdout_7525.csv fusion_out/cure_imprints_holdout.csv
cp fusion_out/ogyei_imprints_7525.csv        fusion_out/ogyei_imprints.csv

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

step rebuttal_stats2 python -u rebuttal_stats2.py
step finalize        python -u finalize_remaining.py

echo; echo "CPU_CROSS_DONE $(date +%H:%M:%S)"

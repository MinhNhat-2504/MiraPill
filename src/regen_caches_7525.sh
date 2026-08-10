#!/bin/bash
# Phase 2 of the 75/25 switch - all remaining GPU caches with the 7525 adapter.
# Waits for the reader eval (regen_reader.sh) to release the GPU first: its
# final summary writes two "token-recall" rows into _r7525_feas_ft.log.
cd "/d/Project/AI Engineer/MiraPill/src" || exit 1
export PYTHONIOENCODING=utf-8

waited=0
until [ "$(grep -ac 'token-recall' _r7525_feas_ft.log 2>/dev/null)" -ge 2 ]; do
  sleep 60
  waited=$((waited+1))
  if [ $waited -ge 100 ]; then
    echo "FATAL: reader eval not finished after 100 min - aborting phase 2"
    exit 1
  fi
done
echo "GPU free ($(date +%H:%M:%S)), starting phase 2"

step () {  # $1 tag, rest = command
  local tag="$1"; shift
  echo "START $tag $(date +%H:%M:%S)"
  "$@" > "_c7525_$tag.log" 2>&1
  local rc=$?
  echo "END $tag rc=$rc $(date +%H:%M:%S)"
  tail -3 "_c7525_$tag.log"
  [ $rc -ne 0 ] && echo "FATAL: $tag failed" && exit 1
}

cp fusion_out/softgate_result.json fusion_out/softgate_result_9010.json
step softgate python -u run_softgate_7525.py
mv fusion_out/softgate_result.json fusion_out/softgate_result_7525.json
cp fusion_out/softgate_result_9010.json fusion_out/softgate_result.json

step cure_db python -u run_bigstack.py cure_imprint_db \
  --manifest "D:\\Data\\CURE_crops\\manifest.csv" \
  --crops_root "D:\\Data\\CURE_crops" \
  --adapter qwen_imprint_lora_7525 \
  --out "D:\\Data\\CURE_crops\\cure_imprint_db_7525.csv"

step cure_q python -u run_bigstack.py cache_cross_7525 --dataset cure

step ogyei_db python -u run_bigstack.py cure_imprint_db \
  --manifest "D:\\Data\\ogyeiv2_crops\\manifest.csv" \
  --crops_root "D:\\Data\\ogyeiv2_crops" \
  --adapter qwen_imprint_lora_7525 \
  --out "D:\\Data\\ogyeiv2_crops\\ogyei_imprint_db_7525.csv"

step ogyei_q python -u run_bigstack.py cache_cross_7525 --dataset ogyei

echo "PHASE2_CACHES_DONE $(date +%H:%M:%S)"

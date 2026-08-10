#!/bin/bash
# Autonomous queue: waits for multi-seed, then ConvNeXt + soft-gate + OCR, then
# aggregates and restores canonical predictions. Each step guarded.
set +e
cd "$(dirname "$0")"
DR="D:/Data/ePillID_data/ePillID_data"
V3="pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv"
V4="pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for multi-seed to finish..."
while ! grep -q "ALL SEEDS DONE" _multiseed.log 2>/dev/null; do sleep 120; done
log "multi-seed done."

# 1) ConvNeXt-Tiny backbone on fold 3 (batch16 for 8GB safety)
log "=== ConvNeXt training start ==="
PYTHONIOENCODING=utf-8 python run_visual_train.py --data_root_dir "$DR" \
  --network convnext_tiny --val_fold "$V3" --test_fold "$V4" \
  --max_epochs 100 --batch_size 16 \
  --results_dir "$(pwd)/visual_results_convnext" > _convnext.log 2>&1
log "ConvNeXt training exit=$?"

# 2) Soft-gate (reader logprob, GPU)
log "=== soft-gate start ==="
PYTHONIOENCODING=utf-8 python softgate_eval.py > _softgate.log 2>&1
log "soft-gate exit=$?"

# 3) OCR baseline (PP-OCRv3 via fixed PaddleReader)
log "=== OCR start ==="
PYTHONIOENCODING=utf-8 python imprint_feasibility.py --readers paddle --by_pill 50 --seed 42 \
  --gt_csv "D:/Data/pillbox_kaggle/Pillbox.csv" --data_root_dir "$DR" \
  --img_dir classification_data \
  --all_imgs_csv "folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv" \
  > _ocr.log 2>&1
# parse paddle token-recall (reference / consumer) into JSON
PYTHONIOENCODING=utf-8 python - <<'PY'
import re, json
try:
    t = open("_ocr.log", encoding="utf-8", errors="ignore").read()
    # lines like: "  reference ... token-recall= 12.3%"
    ref = re.search(r"reference.*?token-recall=\s*([\d.]+)", t)
    con = re.search(r"consumer.*?token-recall=\s*([\d.]+)", t)
    if ref and con:
        json.dump({"engine": "PP-OCRv3", "reference": float(ref.group(1)),
                   "consumer": float(con.group(1))}, open("fusion_out/ocr_result.json", "w"), indent=2)
        print("OCR parsed:", ref.group(1), con.group(1))
    else:
        print("OCR parse: recall lines not found")
except Exception as e:
    print("OCR parse fail:", e)
PY
log "OCR done"

# 4) Restore canonical resnet50 fold-3 + CURE predictions (runs above clobbered outputs/)
cp _canonical_backup/eval_predictions_pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv outputs/ 2>/dev/null
cp _canonical_backup/eval_predictions_cure_val.csv outputs/ 2>/dev/null
log "restored canonical predictions"

# 5) Aggregate everything
PYTHONIOENCODING=utf-8 python finalize_remaining.py > _finalize.log 2>&1
log "finalize exit=$?"
echo "REMAINING EXPERIMENTS DONE" > _remaining_done.marker
log "=== ALL REMAINING EXPERIMENTS DONE ==="

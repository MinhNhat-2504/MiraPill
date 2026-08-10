#!/bin/bash
# Phase 3a: ePillID-only CPU re-derivation with the 75/25 reader cache.
cd "/d/Project/AI Engineer/MiraPill/src" || exit 1
export PYTHONIOENCODING=utf-8

need () { [ -f "$1" ] || { echo "FATAL: missing $1"; exit 1; }; }
need fusion_out/imprints_holdout_7525.csv
need fusion_out/softgate_logprob_7525.csv
need fusion_out/softgate_result_7525.json

cp fusion_out/imprints_holdout_7525.csv fusion_out/imprints_holdout.csv
cp fusion_out/softgate_logprob_7525.csv fusion_out/softgate_logprob.csv
cp fusion_out/softgate_result_7525.json fusion_out/softgate_result.json

step () {
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
step rebuttal_stats python -u rebuttal_stats.py

echo; echo "CPU_EPILLID_DONE $(date +%H:%M:%S)"

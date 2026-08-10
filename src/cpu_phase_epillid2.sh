#!/bin/bash
# Phase 3a resume: e2..e5 + rebuttal_stats (e1 already done).
cd "/d/Project/AI Engineer/MiraPill/src" || exit 1
export PYTHONIOENCODING=utf-8

step () {
  local tag="$1"; shift
  echo; echo "===== $tag $(date +%H:%M:%S) ====="
  "$@" 2>&1 | tee "_p3_$tag.log"
  local rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && echo "FATAL: $tag rc=$rc" && exit 1
}

step e2_signif    python -u run_with_newbase.py e2_significance
step e3_failure   python -u run_with_newbase.py e3_failure_analysis
step e4_openset   python -u run_with_newbase.py e4_openset
step e5_ablation  python -u run_with_newbase.py e5_ablation
step rebuttal_stats python -u rebuttal_stats.py

echo; echo "CPU_EPILLID_DONE $(date +%H:%M:%S)"

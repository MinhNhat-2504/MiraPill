# -*- coding: utf-8 -*-
"""
Appendix Table 24 (hard vs soft gate) with the 75/25 reader.

softgate_eval.py hardcodes adapter="qwen_imprint_lora" and its cache path;
patch both, then run its main() on a big-stack thread (the Qwen vision tower
overflows the default 1 MB stack on Windows). main() still writes
fusion_out/softgate_result.json - the caller must back up the 90/10 copy
first and rename the output to softgate_result_7525.json afterwards.
"""
import sys
import threading

STACK_BYTES = 128 * 1024 * 1024

import imprint_feasibility as impf

_OrigReader = impf.QwenReader


def _Reader7525(model, device, adapter=None):
    print(f"[patch] QwenReader adapter {adapter!r} -> 'qwen_imprint_lora_7525'")
    return _OrigReader(model, device, adapter="qwen_imprint_lora_7525")


impf.QwenReader = _Reader7525

import softgate_eval as se

se.CACHE = "fusion_out/softgate_logprob_7525.csv"

box = {}


def run():
    try:
        se.main()
    except BaseException as exc:
        box["exc"] = exc


threading.stack_size(STACK_BYTES)
t = threading.Thread(target=run)
t.start()
t.join()
if "exc" in box:
    raise box["exc"]
print("[softgate7525] done")

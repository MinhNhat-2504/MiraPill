# -*- coding: utf-8 -*-
"""
Windows launcher for finetune_qwen_imprint.py.

On Windows the cuDNN algorithm search inside the Qwen2-VL vision tower's patch
convolution overflows the default 1 MB thread stack ("Windows fatal exception:
stack overflow"). Running main() on a thread with a large stack fixes it. All
arguments are forwarded verbatim to finetune_qwen_imprint.main().

    python run_ft_bigstack.py --data_dir imprint_ft_data_7525 \
        --out_dir qwen_imprint_lora_7525 --epochs 2
"""
import sys
import threading

import finetune_qwen_imprint

STACK_BYTES = 128 * 1024 * 1024  # Windows caps threading.stack_size at 256 MB


def main():
    threading.stack_size(STACK_BYTES)
    box = {}

    def run():
        try:
            finetune_qwen_imprint.main()
        except BaseException as exc:  # re-raise on the main thread
            box["exc"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "exc" in box:
        raise box["exc"]


if __name__ == "__main__":
    print(f"[launcher] thread stack = {STACK_BYTES // (1024 * 1024)} MB", flush=True)
    sys.exit(main())

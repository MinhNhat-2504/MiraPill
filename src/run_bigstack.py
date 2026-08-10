# -*- coding: utf-8 -*-
"""
Generic Windows launcher: run any module's main() on a large-stack thread.

On Windows the cuDNN algorithm search inside the Qwen2-VL vision-tower patch
convolution overflows the default 1 MB thread stack ("Windows fatal exception:
stack overflow"), which kills both fine-tuning and generation. Running main()
on a thread with a 128 MB stack fixes it.

    python run_bigstack.py imprint_feasibility --readers qwen --by_pill 150 ...
"""
import importlib
import sys
import threading

STACK_BYTES = 128 * 1024 * 1024  # Windows caps threading.stack_size at 256 MB


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python run_bigstack.py <module> [args...]")
    mod_name = sys.argv[1]
    sys.argv = [mod_name + ".py"] + sys.argv[2:]   # forward args verbatim
    mod = importlib.import_module(mod_name)

    threading.stack_size(STACK_BYTES)
    box = {}

    def run():
        try:
            mod.main()
        except BaseException as exc:               # re-raise on the main thread
            box["exc"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "exc" in box:
        raise box["exc"]


if __name__ == "__main__":
    print(f"[launcher] {sys.argv[1:2]} on a {STACK_BYTES // (1024 * 1024)} MB stack",
          flush=True)
    main()

# -*- coding: utf-8 -*-
"""
Run fusion_rerank with the stored query image paths remapped to a new data root.

The saved eval_predictions_*.csv files embed ABSOLUTE image paths from the
machine that produced them (D:\\ePillID_data\\...). If the dataset now lives
elsewhere, every read fails; cache_imprints() swallows the per-image exception,
so the run "succeeds" but writes an all-empty imprint cache and the fusion then
reports exactly the visual-only numbers (+0.00 pp). Remap the paths instead.

    python run_cache_remap.py --old_root "D:/ePillID_data" --new_root "D:/Data/ePillID_data" \
        -- --pred_csv ... --adapter ... --cache_imprints --imprint_cache ...

Everything after the bare -- is forwarded verbatim to fusion_rerank.main().
"""
import argparse
import sys
import threading

STACK_BYTES = 128 * 1024 * 1024


def norm(p):
    return str(p).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_root", required=True)
    ap.add_argument("--new_root", required=True)
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    import fusion_rerank
    old, new = norm(args.old_root).rstrip("/"), norm(args.new_root).rstrip("/")
    orig_parse = fusion_rerank.parse_paths
    stats = {"n": 0, "hit": 0}

    def patched(s):
        out = []
        for p in orig_parse(s):
            q = norm(p)
            stats["n"] += 1
            if q.startswith(old + "/"):
                q = new + q[len(old):]
                stats["hit"] += 1
            out.append(q)
        return out

    fusion_rerank.parse_paths = patched
    sys.argv = ["fusion_rerank.py"] + rest

    threading.stack_size(STACK_BYTES)
    box = {}

    def run():
        try:
            fusion_rerank.main()
        except BaseException as exc:
            box["exc"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join()
    print(f"[remap] rewrote {stats['hit']}/{stats['n']} query image paths "
          f"({old} -> {new})", flush=True)
    if "exc" in box:
        raise box["exc"]


if __name__ == "__main__":
    main()

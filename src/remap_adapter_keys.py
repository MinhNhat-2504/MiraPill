# -*- coding: utf-8 -*-
"""
Rename Qwen2-VL LoRA adapter keys between Transformers module layouts.

Across Transformers releases the Qwen2-VL language tower changed module path.
Adapters saved by a newer release use

    base_model.model.model.language_model.layers.<k>....

while older releases expect

    base_model.model.model.layers.<k>....

When the names do not match, PEFT attaches no weights AND RAISES NO ERROR: the
adapter silently behaves like the un-adapted base model, so a fine-tuned reader
reproduces the baseline numbers. This script rewrites the keys either way.

    # newer-layout adapter -> older (flat) layout
    python remap_adapter_keys.py qwen_imprint_lora qwen_imprint_lora_flat --to flat

    # older (flat) adapter -> newer (nested) layout
    python remap_adapter_keys.py qwen_imprint_lora_7525 out_nested --to nested

Verify afterwards: a correctly attached adapter changes the model's output; if
imprint accuracy still matches the un-adapted baseline, the keys are still wrong.
"""
import argparse
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

NESTED = ".model.language_model.layers."
FLAT = ".model.layers."
SIDECARS = ("adapter_config.json", "tokenizer.json", "tokenizer_config.json",
            "chat_template.jinja", "processor_config.json", "special_tokens_map.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="adapter directory to read")
    ap.add_argument("dst", help="adapter directory to write")
    ap.add_argument("--to", choices=["flat", "nested"], required=True,
                    help="'flat' for older Transformers, 'nested' for newer")
    args = ap.parse_args()

    src_w = os.path.join(args.src, "adapter_model.safetensors")
    if not os.path.exists(src_w):
        raise SystemExit(f"no adapter_model.safetensors in {args.src}")

    old, new = (NESTED, FLAT) if args.to == "flat" else (FLAT, NESTED)
    with safe_open(src_w, framework="pt") as f:
        tensors, renamed = {}, 0
        for k in f.keys():
            nk = k.replace(old, new) if old in k else k
            renamed += nk != k
            tensors[nk] = f.get_tensor(k)

    if renamed == 0:
        print(f"[warn] no key contained '{old}'; the adapter may already be in "
              f"'{args.to}' layout. Writing an unchanged copy.")

    os.makedirs(args.dst, exist_ok=True)
    save_file(tensors, os.path.join(args.dst, "adapter_model.safetensors"))
    for fn in SIDECARS:
        p = os.path.join(args.src, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.dst, fn))
    print(f"[done] {renamed}/{len(tensors)} keys renamed -> {args.dst}")


if __name__ == "__main__":
    main()

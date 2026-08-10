# -*- coding: utf-8 -*-
"""
Experiment A, step 2 — LoRA fine-tune Qwen2-VL-2B as a pill imprint reader.

Input: train.jsonl / val.jsonl from prep_imprint_data.py
  each line: {"images": [front, back], "target": "G 148", "label_prod_code": ...}

Two images (front+back) -> full imprint string. LoRA on the language-model
projections only; vision tower frozen. Fits an 8 GB GPU with bs=1 + grad
accumulation + gradient checkpointing (fp16). Saves the adapter to --out_dir.

Setup:
    pip install peft

Smoke test (a few steps, confirm it runs):
    python finetune_qwen_imprint.py --data_dir imprint_ft_data --out_dir qwen_imprint_lora \\
        --max_steps 3 --smoke

Full run:
    python finetune_qwen_imprint.py --data_dir imprint_ft_data --out_dir qwen_imprint_lora \\
        --epochs 2
"""
import argparse
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROMPT = ("These are the two faces (front and back) of ONE pill. Transcribe ALL "
          "imprinted characters across both faces: letters and numbers, including "
          "faint/debossed text. Output ONLY the characters, uppercase, space-"
          "separated. If a face is blank, ignore it.")


def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


class ImprintDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def build_collator(processor):
    im_end = "<|im_end|>"

    def collate(batch):
        # batch size is 1 (we accumulate gradients) to keep image handling simple
        r = batch[0]
        imgs = [Image.open(p).convert("RGB") for p in r["images"]]
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": PROMPT}]
        messages = [{"role": "user", "content": content}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + r["target"] + im_end + "\n"

        full = processor(text=[full_text], images=imgs, return_tensors="pt")
        prompt = processor(text=[prompt_text], images=imgs, return_tensors="pt")
        lp = prompt["input_ids"].shape[1]

        labels = full["input_ids"].clone()
        labels[:, :lp] = -100                      # mask the prompt (incl. image tokens)
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[full["input_ids"] == pad_id] = -100
        full["labels"] = labels
        return full

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="imprint_ft_data")
    ap.add_argument("--out_dir", default="qwen_imprint_lora")
    ap.add_argument("--base_model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = full epochs")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, sanity only")
    ap.add_argument("--save_every", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2VLForConditionalGeneration as VLModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as VLModel
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(
        args.base_model, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)
    model = VLModel.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map=device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA on the language-model attention/MLP projections; freeze the vision tower
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = load_jsonl(os.path.join(args.data_dir, "train.jsonl"),
                      limit=16 if args.smoke else None)
    ds = ImprintDataset(rows)
    dl = DataLoader(ds, batch_size=1, shuffle=True,
                    collate_fn=build_collator(processor))

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr)
    total_steps = args.max_steps or int(len(dl) * args.epochs)
    print(f"[train] samples={len(ds)} optim-steps~={total_steps} "
          f"(grad_accum={args.grad_accum})")

    model.train()
    step, micro, running = 0, 0, 0.0
    done = False
    for epoch in range(10_000):
        if done:
            break
        for batch in dl:
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()
            micro += 1
            if micro % args.grad_accum == 0:
                optim.step()
                optim.zero_grad()
                step += 1
                if step % 10 == 0 or args.smoke:
                    print(f"  epoch {epoch} step {step}/{total_steps} "
                          f"loss={running / micro:.4f}")
                if args.save_every and step % args.save_every == 0:
                    model.save_pretrained(args.out_dir)
                if args.max_steps and step >= args.max_steps:
                    done = True
                    break
        if not args.max_steps and epoch + 1 >= args.epochs:
            done = True

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"[train] DONE. LoRA adapter saved -> {args.out_dir}")
    print(f"[eval]  evaluate with:\n"
          f"  python imprint_feasibility.py --readers qwen --by_pill 50 \\\n"
          f"    --adapter \"{args.out_dir}\" --gt_csv <Pillbox.csv> --data_root_dir <...>")


if __name__ == "__main__":
    main()

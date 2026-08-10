"""
Imprint-reader feasibility test (E5-mini) for the MIRA-Pill Q1 plan.

Goal: before investing in the full multimodal pipeline, measure how well an
off-the-shelf imprint reader can recover the imprinted characters on ePillID
pills. We compare PaddleOCR (PP-OCRv3) and Qwen2-VL-2B against the ground-truth
`SPLIMPRINT` text from the NIH Pillbox masterdata, separately for reference
(studio) and consumer (real-world) images.

Decision gate:
    reference read-accuracy >= 70%  AND  consumer >= 40%   -> proceed with confidence
    otherwise                                               -> fine-tune a VLM imprint
                                                               reader (use SPLIMPRINT as
                                                               weak labels on reference imgs)

This script has NO dependency on the rest of the repo. It reads the fold CSV and
the Pillbox masterdata directly, joins them on `label_prod_code`, samples N
reference + N consumer images that actually have an imprint, runs the selected
reader(s), and reports exact-match / token-set / character-error-rate per domain.

Run order:
    1) python imprint_feasibility.py --dry_run            # verify data join + sampling
    2) python imprint_feasibility.py --readers paddle     # cheap, CPU-friendly
    3) python imprint_feasibility.py --readers qwen        # needs ~5GB VRAM (RTX 4060 OK)
    4) python imprint_feasibility.py --readers both
"""

import argparse
import json
import os
import re
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image


# --------------------------------------------------------------------------- #
#  String normalization + metrics
# --------------------------------------------------------------------------- #
# Pillbox placeholder/non-visual tokens that no OCR can read (strip from GT)
_GT_NOISE_TOKENS = {"LOGO", "SYMBOL", "STYLIZED", "DESIGN"}


def normalize_imprint(s):
    """Uppercase, treat ';'/'|' (side separators) as spaces, keep only
    alphanumerics + spaces, collapse whitespace."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).upper()
    s = s.replace(";", " ").replace("|", " ").replace("/", " ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_gt_imprint(s):
    """Remove Pillbox placeholder tokens (e.g. 'LOGO') that denote a graphic
    symbol rather than readable text, so they don't unfairly penalize a reader."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    parts = re.split(r"([;|])", str(s))  # keep the side separators
    kept = []
    for p in parts:
        if p in (";", "|"):
            kept.append(p)
        else:
            toks = [t for t in p.split() if t.upper() not in _GT_NOISE_TOKENS]
            kept.append(" ".join(toks))
    out = "".join(kept)
    out = re.sub(r"[;|]\s*[;|]", ";", out)        # collapse empty side groups
    return out.strip("; |")


def preprocess_image(pil_img, mode):
    """Optional image enhancement before the reader sees the pill.
    'clahe'  : boost local contrast (helps faint debossed text)
    'upscale': 2x LANCZOS upscale (small 224 segmented crops -> 448)
    'both'   : upscale then CLAHE"""
    if mode in (None, "none"):
        return pil_img
    import cv2
    arr = np.array(pil_img.convert("RGB"))
    if mode in ("upscale", "both"):
        h, w = arr.shape[:2]
        arr = cv2.resize(arr, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    if mode in ("clahe", "both"):
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l), a, b))
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(arr)


def levenshtein(a, b):
    """Character-level edit distance (no external deps)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def score_one(pred, gt):
    """Return dict of match metrics for a single (prediction, ground-truth) pair.

    ePillID images are SINGLE-sided, but the Pillbox imprint covers BOTH sides
    (separated by ';', e.g. '93;148'). So the headline metric is `side_match`:
    a hit if the reading equals the full imprint OR any one side-group of it."""
    p, g = normalize_imprint(pred), normalize_imprint(gt)
    p_ns, g_ns = p.replace(" ", ""), g.replace(" ", "")

    # side-groups: split the RAW gt on side/line separators, then normalize each
    groups = [normalize_imprint(x)
              for x in re.split(r"[;|]", str(gt)) if normalize_imprint(x)]

    exact = int(p == g and g != "")                      # full imprint, both sides
    # side_match: read the visible side correctly (or the whole pill)
    side_match = int(p != "" and (p == g or p in groups))
    # token-level recall: fraction of GT tokens recovered (graceful partial credit)
    g_tok, p_tok = g.split(), set(p.split())
    token_recall = (sum(t in p_tok for t in g_tok) / len(g_tok)) if g_tok else 0.0
    # token-set: EVERY ground-truth token present (strict, both sides)
    token_set = int(set(g_tok).issubset(p_tok) and len(g_tok) > 0)
    # CER on space-removed strings (1.0 = totally wrong, 0.0 = perfect)
    cer = levenshtein(p_ns, g_ns) / max(len(g_ns), 1)
    contains = int(g_ns != "" and g_ns in p_ns)

    return dict(exact=exact, side_match=side_match, token_recall=token_recall,
                token_set=token_set, cer=cer, contains=contains,
                pred_norm=p, gt_norm=g)


# --------------------------------------------------------------------------- #
#  Ground-truth imprint join (fold CSV  x  Pillbox masterdata)
# --------------------------------------------------------------------------- #
def product_code_to_label_prod_code(pc):
    """Replicates classif_utils.add_prodlbl_id_cols + add_label_prod_code."""
    try:
        parts = str(pc).split("-")
        label_code_id = int(parts[0])
        second = parts[1]
        prod_code_id = int(second[1:] if second[:1] == "N" else second)
        return f"{label_code_id}-{prod_code_id}"
    except Exception:
        return None


def load_ground_truth_imprints(gt_path):
    """Return DataFrame[label_prod_code, splimprint] from a Pillbox masterdata
    file (TSV/.tab) OR any simple CSV/TSV that exposes an imprint column plus
    either a product_code or a label_prod_code column.

    Column matching is case-insensitive and accepts common aliases."""
    sep = "\t" if gt_path.lower().endswith((".tab", ".tsv")) else None  # None = sniff
    df = None
    for enc in ("cp1252", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(gt_path, sep=sep, dtype=str, encoding=enc,
                             engine="python", on_bad_lines="skip")
            break
        except (UnicodeDecodeError, Exception):
            continue
    if df is None:
        raise ValueError(f"could not read ground-truth file: {gt_path}")

    cols = {c.upper().strip(): c for c in df.columns}
    imp_col = next((cols[k] for k in ("SPLIMPRINT", "IMPRINT") if k in cols), None)
    if imp_col is None:
        raise ValueError(f"GT file has no imprint column (SPLIMPRINT/IMPRINT). "
                         f"Found: {list(df.columns)[:25]}")

    out = pd.DataFrame()
    out["splimprint"] = df[imp_col].apply(clean_gt_imprint)  # strip LOGO etc.

    # build the join key: prefer explicit label_prod_code, else product_code,
    # else (label_code_id, prod_code_id)
    if "LABEL_PROD_CODE" in cols:
        out["label_prod_code"] = df[cols["LABEL_PROD_CODE"]].astype(str)
    elif "PRODUCT_CODE" in cols:
        out["label_prod_code"] = df[cols["PRODUCT_CODE"]].apply(product_code_to_label_prod_code)
    elif "LABEL_CODE_ID" in cols and "PROD_CODE_ID" in cols:
        out["label_prod_code"] = (df[cols["LABEL_CODE_ID"]].astype(str) + "-"
                                  + df[cols["PROD_CODE_ID"]].astype(str))
    else:
        raise ValueError("GT file has no usable key (need PRODUCT_CODE, or "
                         "LABEL_PROD_CODE, or LABEL_CODE_ID+PROD_CODE_ID). "
                         f"Found: {list(df.columns)[:25]}")

    out = out.dropna(subset=["label_prod_code", "splimprint"])
    out = out[out["splimprint"].apply(lambda s: normalize_imprint(s) != "")]
    out = out.drop_duplicates(subset=["label_prod_code"], keep="first")
    return out[["label_prod_code", "splimprint"]]


def build_sample(args):
    fold_csv = args.all_imgs_csv
    if not os.path.isabs(fold_csv):
        fold_csv = os.path.join(args.data_root_dir, fold_csv)
    if not os.path.exists(fold_csv):
        sys.exit(f"[ERROR] fold CSV not found: {fold_csv}\n"
                 f"        Fix --data_root_dir / --all_imgs_csv.")

    df = pd.read_csv(fold_csv)
    if "image_path" not in df.columns:
        if "images" in df.columns:
            df["image_path"] = df["images"]
        else:
            sys.exit(f"[ERROR] fold CSV has no image_path/images column. Cols: {list(df.columns)}")
    if "is_ref" not in df.columns:
        sys.exit(f"[ERROR] fold CSV has no is_ref column. Cols: {list(df.columns)}")
    df["is_ref"] = df["is_ref"].astype(bool)

    # build the imprint-DB join key from whatever the fold CSV provides
    if "label_prod_code" not in df.columns:
        if {"label_code_id", "prod_code_id"}.issubset(df.columns):
            df["label_prod_code"] = (df["label_code_id"].astype(str) + "-"
                                     + df["prod_code_id"].astype(str))
        else:
            sys.exit(f"[ERROR] fold CSV lacks label_prod_code and "
                     f"label_code_id/prod_code_id. Cols: {list(df.columns)}")
    df["label_prod_code"] = df["label_prod_code"].astype(str)

    full = lambda rel: os.path.join(args.data_root_dir, args.img_dir, rel)
    df["full_path"] = df["image_path"].apply(full)
    df = df[df["full_path"].apply(os.path.exists)]

    if args.no_gt:
        # eyeball mode: no ground-truth needed, just sample images & dump preds
        df["splimprint"] = ""
        merged = df
        print(f"[no_gt] images on disk -> reference: {int(df['is_ref'].sum())}  "
              f"consumer: {int((~df['is_ref']).sum())}  (no imprint GT; preds only)")
    else:
        if args.gt_csv:
            # user-provided GT file: resolve relative to the current directory
            gt_path = args.gt_csv
        else:
            # masterdata default lives under data_root/img_dir/resources/
            gt_path = args.masterdata
            if not os.path.isabs(gt_path):
                gt_path = os.path.join(args.data_root_dir, args.img_dir, gt_path)
        if not os.path.exists(gt_path):
            sys.exit(
                f"[ERROR] imprint ground-truth file not found: {gt_path}\n"
                f"        The ePillID data release does NOT bundle it. Options:\n"
                f"        (1) --make_gt_template 150  -> hand-label a sample from clean\n"
                f"            reference images (no download needed), then pass --gt_csv\n"
                f"        (2) get the Pillbox masterdata (Kaggle dhuh137/pillbox, etc.),\n"
                f"            then pass it via --gt_csv <path>\n"
                f"        (3) --no_gt  -> just eyeball reader output without scoring")
        gt = load_ground_truth_imprints(gt_path)
        merged = df.merge(gt, on="label_prod_code", how="inner")
        merged = merged[merged["splimprint"].apply(lambda s: normalize_imprint(s) != "")]
        print(f"[join] rows with imprint GT  -> reference: "
              f"{int(merged['is_ref'].sum())}  consumer: {int((~merged['is_ref']).sum())}")

    if args.restrict_types:
        # restrict the eval pool to a given set of pill types (e.g. held-out VAL
        # types) for a clean "unseen types" evaluation
        wanted = set()
        with open(args.restrict_types, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    wanted.add(str(json.loads(line)["label_prod_code"]))
                else:
                    wanted.add(line.split(",")[0])
        before = len(merged)
        merged = merged[merged["label_prod_code"].isin(wanted)]
        print(f"[restrict] kept {merged['label_prod_code'].nunique()} types "
              f"({len(merged)}/{before} imgs) from {args.restrict_types}")

    ref = merged[merged["is_ref"]]
    cons = merged[~merged["is_ref"]]

    keep_cols = ["full_path", "domain", "label_prod_code", "splimprint"]
    if "is_front" in merged.columns:
        keep_cols.append("is_front")

    if args.by_pill > 0:
        # BOTH-SIDES eval: pick K pill types per domain, take ALL their side
        # images (capped), so a pill's imprint can be pooled across both faces.
        def sample_pills(sub, k):
            if len(sub) == 0 or k <= 0:
                return sub.iloc[0:0]
            types = pd.Series(sub["label_prod_code"].unique())
            types = types.sample(min(k, len(types)), random_state=args.seed)
            chosen = sub[sub["label_prod_code"].isin(set(types))]
            # cap images per pill to bound inference cost
            return chosen.groupby("label_prod_code", group_keys=False).head(
                args.max_per_pill)

        ref_s = sample_pills(ref, args.by_pill).assign(domain="reference")
        cons_s = sample_pills(cons, args.by_pill).assign(domain="consumer")
        sample_df = pd.concat([ref_s, cons_s]).reset_index(drop=True)
        print(f"[by_pill] reference: {ref_s['label_prod_code'].nunique()} pills / "
              f"{len(ref_s)} imgs | consumer: {cons_s['label_prod_code'].nunique()} "
              f"pills / {len(cons_s)} imgs")
        return sample_df[keep_cols]

    def sample(sub, n):
        if len(sub) == 0 or n <= 0:
            return sub.iloc[0:0]
        # one image per pill type first (diversity), then top up
        uniq = sub.groupby("label_prod_code", group_keys=False).sample(
            1, random_state=args.seed)
        if len(uniq) >= n:
            return uniq.sample(n, random_state=args.seed)
        rest = sub.drop(uniq.index)
        extra = rest.sample(min(n - len(uniq), len(rest)), random_state=args.seed)
        return pd.concat([uniq, extra])

    ref_s = sample(ref, args.n_ref).assign(domain="reference")
    cons_s = sample(cons, args.n_cons).assign(domain="consumer")
    sample_df = pd.concat([ref_s, cons_s]).reset_index(drop=True)
    print(f"[sample] selected reference: {len(ref_s)}  consumer: {len(cons_s)}")
    return sample_df[keep_cols]


def _load_fold_with_keys(args):
    """Read the fold CSV and attach label_prod_code + full_path (no GT needed)."""
    fold_csv = args.all_imgs_csv
    if not os.path.isabs(fold_csv):
        fold_csv = os.path.join(args.data_root_dir, fold_csv)
    if not os.path.exists(fold_csv):
        sys.exit(f"[ERROR] fold CSV not found: {fold_csv}")
    df = pd.read_csv(fold_csv)
    if "image_path" not in df.columns and "images" in df.columns:
        df["image_path"] = df["images"]
    df["is_ref"] = df["is_ref"].astype(bool)
    if "label_prod_code" not in df.columns:
        df["label_prod_code"] = (df["label_code_id"].astype(str) + "-"
                                 + df["prod_code_id"].astype(str))
    df["label_prod_code"] = df["label_prod_code"].astype(str)
    full = lambda rel: os.path.join(args.data_root_dir, args.img_dir, rel)
    df["full_path"] = df["image_path"].apply(full)
    return df[df["full_path"].apply(os.path.exists)]


def make_gt_template(args):
    """Export K clean reference images + a blank CSV so a human can hand-label
    the imprint by reading the studio reference images. Output GT then applies to
    BOTH reference and consumer images of each pill type (same imprint per type).
    This unblocks the feasibility GATE with no masterdata download."""
    import shutil
    df = _load_fold_with_keys(args)
    # keep only pill types that have a reference AND a consumer image,
    # so the hand-labelled GT can score both domains
    has_ref = set(df[df["is_ref"]]["label_prod_code"])
    has_cons = set(df[~df["is_ref"]]["label_prod_code"])
    both = has_ref & has_cons
    ref_df = df[df["is_ref"] & df["label_prod_code"].isin(both)]
    if len(ref_df) == 0:
        sys.exit("[ERROR] no pill types with both reference+consumer images found.")

    picks = ref_df.groupby("label_prod_code", group_keys=False).sample(
        1, random_state=args.seed)
    k = min(args.make_gt_template, len(picks))
    picks = picks.sample(k, random_state=args.seed).reset_index(drop=True)

    img_dir = os.path.join(args.out_dir, "gt_images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for _, r in picks.iterrows():
        dst_name = f"{r['label_prod_code']}.jpg"
        try:
            shutil.copy(r["full_path"], os.path.join(img_dir, dst_name))
        except Exception as e:
            print(f"  [warn] could not copy {r['full_path']}: {e}")
        rows.append({"label_prod_code": r["label_prod_code"],
                     "ref_image": dst_name, "splimprint": ""})
    tmpl = pd.DataFrame(rows)
    tmpl_path = os.path.join(args.out_dir, "gt_template.csv")
    tmpl.to_csv(tmpl_path, index=False)
    print(f"\n[make_gt_template] Exported {len(picks)} reference images -> {img_dir}")
    print(f"[make_gt_template] Blank template -> {tmpl_path}")
    print("  HOW TO LABEL: open each image in gt_images/, read the imprinted text,")
    print("  and type it into the 'splimprint' column (use ';' to separate the two")
    print("  sides/lines, e.g. 'G;10'). Leave blank if the pill has no imprint.")
    print("  THEN score readers with:")
    print(f"    python imprint_feasibility.py --readers both --gt_csv \"{tmpl_path}\" "
          f"--data_root_dir \"{args.data_root_dir}\"")


# --------------------------------------------------------------------------- #
#  Readers
# --------------------------------------------------------------------------- #
class PaddleReader:
    name = "paddle"

    def __init__(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            sys.exit("[ERROR] paddleocr not installed.\n"
                     "        pip install paddlepaddle paddleocr")
        # Pin PP-OCRv3 explicitly (paddleocr 3.x supports ocr_version); fall back
        # across API generations.
        for kw in (dict(lang="en", ocr_version="PP-OCRv3", use_textline_orientation=True),
                   dict(lang="en", ocr_version="PP-OCRv3"),
                   dict(use_angle_cls=True, lang="en", show_log=False),
                   dict(lang="en")):
            try:
                self.ocr = PaddleOCR(**kw)
                break
            except TypeError:
                continue

    def read(self, pil_img):
        arr = np.array(pil_img.convert("RGB"))
        try:
            res = self.ocr.ocr(arr)
        except Exception:
            try:
                res = self.ocr.predict(arr)
            except Exception:
                res = []
        texts = []
        try:
            for page in (res or []):
                # paddleocr 3.x: page is a dict/OCRResult with 'rec_texts'
                if isinstance(page, dict) and "rec_texts" in page:
                    texts.extend(str(t) for t in page["rec_texts"] if t)
                    continue
                rt = getattr(page, "get", lambda *_: None)("rec_texts") if hasattr(page, "get") else None
                if rt:
                    texts.extend(str(t) for t in rt if t)
                    continue
                # legacy format: list of [box, (text, score)]
                for line in (page or []):
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        info = line[1]
                        txt = info[0] if isinstance(info, (list, tuple)) else info
                        if txt:
                            texts.append(str(txt))
        except Exception:
            pass
        return " ".join(texts)


class TesseractReader:
    """Classical OCR baseline for Table 5 (tab:ocrbase). Uses tesserocr (which
    bundles the Tesseract 5 engine, no system install needed) with eng
    traineddata from ./tessdata. PSM SPARSE_TEXT suits isolated debossed
    imprints better than the default block-of-text assumption; falls back to
    SINGLE_BLOCK when the sparse pass reads nothing."""
    name = "tesseract"

    def __init__(self):
        try:
            import tesserocr
        except ImportError:
            sys.exit("[ERROR] tesserocr not installed.\n"
                     "        pip install tesserocr (Windows: GitHub wheel)")
        tessdata = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tessdata")
        if not os.path.exists(os.path.join(tessdata, "eng.traineddata")):
            sys.exit(f"[ERROR] eng.traineddata not found in {tessdata}")
        self.tesserocr = tesserocr
        self.api_sparse = tesserocr.PyTessBaseAPI(path=tessdata,
                                                  psm=tesserocr.PSM.SPARSE_TEXT)
        self.api_block = tesserocr.PyTessBaseAPI(path=tessdata,
                                                 psm=tesserocr.PSM.SINGLE_BLOCK)
        print(f"[tesseract] {tesserocr.tesseract_version().splitlines()[0]}"
              f" | tessdata: {tessdata}")

    def read(self, pil_img):
        img = pil_img.convert("RGB")
        txt = ""
        try:
            self.api_sparse.SetImage(img)
            txt = self.api_sparse.GetUTF8Text()
            if not txt.strip():
                self.api_block.SetImage(img)
                txt = self.api_block.GetUTF8Text()
        except Exception:
            txt = ""
        return " ".join(txt.split())


class QwenReader:
    name = "qwen"

    def __init__(self, model_id, device, adapter=None):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2VLForConditionalGeneration as VLModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as VLModel
        self.torch = torch
        dtype = torch.float16 if device == "cuda" else torch.float32
        # cap pixels so a 224-ish pill image stays small on 8GB VRAM
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
        self.model = VLModel.from_pretrained(
            model_id, torch_dtype=dtype,
            device_map=device if device == "cuda" else None)
        if adapter:  # load a fine-tuned LoRA adapter (experiment A)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            print(f"  [qwen] loaded LoRA adapter from {adapter}")
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.prompt = ("This is a close-up of a single pill/tablet/capsule. Almost all "
                       "pills carry imprinted text: letters, numbers or a score line, "
                       "often faint, debossed (pressed in) or low-contrast. Look very "
                       "carefully, including faint or partially lit characters, and "
                       "transcribe EVERY character you can see. Output ONLY the imprint "
                       "characters, uppercase, separated by spaces, with no explanation. "
                       "Make your best attempt even if faint; only output NONE if the "
                       "surface is truly blank.")

    def read(self, pil_img):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self.prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_img.convert("RGB")],
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        ans = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0].strip()
        return "" if ans.upper().startswith("NONE") else ans

    def read_with_logprob(self, pil_img):
        """Return (imprint_string, mean_token_logprob) for the soft-gate study.
        mean_token_logprob is the average log-probability of the greedily
        generated (non-special) tokens; higher = more confident read."""
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self.prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_img.convert("RGB")],
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[:, inputs["input_ids"].shape[1]:]
        # mean log-prob over generated steps
        lps = []
        for step, score in enumerate(out.scores):
            logp = self.torch.log_softmax(score[0], dim=-1)
            tok = seq[0, step]
            lps.append(float(logp[tok]))
        mean_lp = float(sum(lps) / len(lps)) if lps else -1e9
        ans = self.processor.batch_decode(
            seq, skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0].strip()
        imp = "" if ans.upper().startswith("NONE") else ans
        return imp, mean_lp


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def run_reader(reader, sample_df, preprocess="none"):
    rows = []
    n = len(sample_df)
    for i, r in sample_df.iterrows():
        try:
            img = Image.open(r["full_path"])
            img = preprocess_image(img, preprocess)
            pred = reader.read(img)
        except Exception as e:
            pred = ""
            print(f"  [warn] {reader.name} failed on {r['full_path']}: {e}")
        sc = score_one(pred, r["splimprint"])
        rows.append({**r.to_dict(), "reader": reader.name, "pred_raw": pred, **sc})
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  [{reader.name}] {i + 1}/{n}")
    return rows


def summarize(df):
    print("\n" + "=" * 64)
    print("FEASIBILITY SUMMARY  (read-accuracy by reader x domain)")
    print("=" * 64)
    gate = {}
    for reader in sorted(df["reader"].unique()):
        for domain in ["reference", "consumer"]:
            sub = df[(df["reader"] == reader) & (df["domain"] == domain)]
            if len(sub) == 0:
                continue
            side = 100 * sub["side_match"].mean()
            recall = 100 * sub["token_recall"].mean()
            exact = 100 * sub["exact"].mean()
            cer = sub["cer"].mean()
            empty = 100 * (sub["pred_norm"].fillna("").astype(str).str.strip() == "").mean()
            print(f"  {reader:7s} | {domain:9s} | n={len(sub):3d} | "
                  f"side-match={side:5.1f}%  token-recall={recall:5.1f}%  "
                  f"exact={exact:5.1f}%  empty-out={empty:4.1f}%  CER={cer:.3f}")
            gate[(reader, domain)] = side  # side-match = headline metric
    print("=" * 64)
    print("DECISION GATE (side-match accuracy):  reference >= 70%  AND  consumer >= 40%")
    for reader in sorted(df["reader"].unique()):
        ref = gate.get((reader, "reference"), 0)
        cons = gate.get((reader, "consumer"), 0)
        verdict = "PROCEED" if (ref >= 70 and cons >= 40) else "NEEDS VLM FINE-TUNING"
        print(f"  {reader:7s}: ref={ref:5.1f}%  cons={cons:5.1f}%  ->  {verdict}")
    print("=" * 64 + "\n")


def summarize_by_pill(df):
    """BOTH-SIDES eval: pool every reading for a pill (across its side images),
    then score the union against the full imprint. This matches the ePillID
    both-sides protocol and avoids penalizing blank/score-only faces."""
    print("\n" + "=" * 64)
    print("BOTH-SIDES SUMMARY  (imprint pooled across a pill's side images)")
    print("=" * 64)
    gate = {}
    for reader in sorted(df["reader"].unique()):
        for domain in ["reference", "consumer"]:
            sub = df[(df["reader"] == reader) & (df["domain"] == domain)]
            if len(sub) == 0:
                continue
            rows = []
            for lpc, g in sub.groupby("label_prod_code"):
                pooled = " ".join(str(p) for p in g["pred_norm"].fillna("")
                                  if str(p).strip())
                gt = g["gt_norm"].iloc[0]
                sc = score_one(pooled, gt)
                # both-sides hit = every GT token recovered across the two faces
                rows.append({"recall": sc["token_recall"],
                             "complete": sc["token_set"], "n_imgs": len(g)})
            rr = pd.DataFrame(rows)
            complete = 100 * rr["complete"].mean()
            recall = 100 * rr["recall"].mean()
            print(f"  {reader:7s} | {domain:9s} | pills={len(rr):3d} "
                  f"(avg {rr['n_imgs'].mean():.1f} imgs/pill) | "
                  f"complete-imprint={complete:5.1f}%  token-recall={recall:5.1f}%")
            gate[(reader, domain)] = recall
    print("=" * 64)
    print("DECISION GATE (both-sides token-recall):  reference >= 70%  AND  consumer >= 40%")
    for reader in sorted(df["reader"].unique()):
        ref = gate.get((reader, "reference"), 0)
        cons = gate.get((reader, "consumer"), 0)
        verdict = "PROCEED" if (ref >= 70 and cons >= 40) else "NEEDS VLM FINE-TUNING"
        print(f"  {reader:7s}: ref={ref:5.1f}%  cons={cons:5.1f}%  ->  {verdict}")
    print("=" * 64 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root_dir", default="/mydata")
    ap.add_argument("--img_dir", default="classification_data")
    ap.add_argument("--all_imgs_csv",
                    default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/"
                            "pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    ap.add_argument("--masterdata", default="resources/pillbox_201805.tab",
                    help="Pillbox masterdata (TSV/.tab); relative to data_root_dir/img_dir")
    ap.add_argument("--gt_csv", default=None,
                    help="alternative imprint ground-truth file (CSV/TSV) with an "
                         "imprint column + product_code/label_prod_code key; "
                         "overrides --masterdata")
    ap.add_argument("--no_gt", action="store_true",
                    help="run readers WITHOUT ground-truth: sample images, dump "
                         "predictions only (to eyeball reader quality / confirm setup)")
    ap.add_argument("--make_gt_template", type=int, default=0, metavar="K",
                    help="export K clean reference images + a blank CSV for manual "
                         "imprint labelling (unblocks the GATE without masterdata)")
    ap.add_argument("--n_ref", type=int, default=75)
    ap.add_argument("--n_cons", type=int, default=75)
    ap.add_argument("--by_pill", type=int, default=0, metavar="K",
                    help="BOTH-SIDES eval: sample K pill types per domain and pool "
                         "the imprint reading across each pill's side images "
                         "(matches the ePillID both-sides protocol)")
    ap.add_argument("--max_per_pill", type=int, default=4,
                    help="cap on side images per pill in --by_pill mode")
    ap.add_argument("--preprocess", default="none",
                    choices=["none", "clahe", "upscale", "both"],
                    help="optional image enhancement before the reader (experiment B)")
    ap.add_argument("--restrict_types", default=None,
                    help="file (jsonl/txt) of label_prod_codes to restrict eval to "
                         "(e.g. held-out val.jsonl for a clean unseen-types eval)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--readers", default="both",
                    choices=["paddle", "qwen", "both", "tesseract"])
    ap.add_argument("--qwen_model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="path to a fine-tuned LoRA adapter for Qwen (experiment A)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_dir", default="feasibility_out")
    ap.add_argument("--dry_run", action="store_true",
                    help="only build + save the sample, do not load any model")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.make_gt_template > 0:
        make_gt_template(args)
        return

    sample_df = build_sample(args)
    sample_path = os.path.join(args.out_dir, "sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"[saved] sample -> {sample_path}")

    if args.dry_run:
        print("\n[dry_run] Sample preview:")
        print(sample_df.head(10).to_string())
        print("\n[dry_run] Done. Re-run without --dry_run to score readers.")
        return

    readers = []
    if args.readers in ("paddle", "both"):
        readers.append(PaddleReader())
    if args.readers == "tesseract":
        readers.append(TesseractReader())
    if args.readers in ("qwen", "both"):
        readers.append(QwenReader(args.qwen_model, args.device, adapter=args.adapter))

    all_rows = []
    for reader in readers:
        print(f"\n[run] reader = {reader.name}  preprocess = {args.preprocess}")
        all_rows.extend(run_reader(reader, sample_df, args.preprocess))

    res = pd.DataFrame(all_rows)
    res_path = os.path.join(args.out_dir, "per_image_results.csv")
    res.to_csv(res_path, index=False)
    print(f"[saved] per-image results -> {res_path}")
    if args.no_gt:
        print("\n[no_gt] No ground-truth -> scoring skipped. Eyeball predictions:")
        cols = ["domain", "label_prod_code", "reader", "pred_raw"]
        print(res[cols].head(40).to_string())
        print(f"\n[no_gt] Inspect full predictions in {res_path}. "
              f"Provide --gt_csv later to compute the decision gate.")
    elif args.by_pill > 0:
        summarize_by_pill(res)
    else:
        summarize(res)


if __name__ == "__main__":
    main()

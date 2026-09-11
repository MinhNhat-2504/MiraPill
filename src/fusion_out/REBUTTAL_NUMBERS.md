# CMC-87689 Revision — REAL numbers (computed from cached predictions)

## Config (verbatim from source)
- CBP: projDim (sketch dim) = **8192**; embedding d = **2048** (cont_dims); dimension_reduction=256.
- Reader fine-tune: Qwen2-VL-2B, LoRA rank 16, vision tower frozen, 18.4M params (0.83% of 2.2B).
  Optimizer **AdamW**, lr **1e-4**, per-device batch **1**, grad-accum **8** (effective batch 8), **2 epochs**, loss 0.40->0.26, ~3h47m RTX 4060.
- Imprint cleaning rules (normalize_imprint): uppercase; treat `;` `|` `/` as spaces; keep only `[A-Z0-9 ]`; collapse whitespace. GT extra: strip Pillbox non-visual tokens (e.g. "LOGO") via clean_gt_imprint.
- Match token sets = whitespace split after normalization.
- CURE consensus DB: run fine-tuned reader on every clean reference image of a class; **union of all reads, tokens deduplicated (order kept)** — NOT a ">=X of reads" count threshold. IDF then down-weights non-discriminative tokens. (Draft letter wording "retained when appears in >= XX reads" must be corrected to union+dedup.)
- VLM prompt (verbatim): "This is a close-up of a single pill/tablet/capsule. Almost all pills carry imprinted text: letters, numbers or a score line, often faint, debossed (pressed in) or low-contrast. Look very carefully, including faint or partially lit characters, and transcribe EVERY character you can see. Output ONLY the imprint characters, uppercase, separated by spaces, with no explanation. Make your best attempt even if faint; only output NONE if the surface is truly blank."

## Table 2 — data flow (per fold; test fold _4 common to all 4 training runs)
- Train types **4518**, test types **192**, reference gallery images **9036**, consumer test images **745**, both-sides queries **626**, excluded **0** (conservative: none excluded), final test queries **626**. (All four identical: balanced official split, common held-out test fold _4.)
- Pooled over 4 folds = **2504** queries.
- NOTE (integrity): this is 4 models on 4 official validation folds evaluated on a COMMON held-out test fold _4 (the ePillID official protocol), not test-fold rotation. mean±std = training/val variability.

## Table 11 — extended statistics (10,000 bootstrap replicates)
- Class-level clustered bootstrap 95% CI of ΔTop-1: **[2.01, 6.74] pp**
- Fold×class clustered bootstrap 95% CI of ΔTop-1: **[2.81, 5.60] pp**
- Clustered bootstrap 95% CI of ΔTop-5: **[0.08, 0.95] pp**
- Clustered bootstrap 95% CI of ΔMAP: **[1.18, 3.79] pp**
- CURE paired bootstrap 95% CI of ΔTop-1: **[2.87, 10.69] pp** (excludes 0)
- OGYEIv2 paired bootstrap 95% CI of ΔTop-1: **[-0.63, 6.04] pp** (INCLUDES 0 — not significant; consistent with "no aggregate degradation")
- Fine-tuned vs off-the-shelf reader (fold 3) McNemar: χ²=**4.0**, **p=0.0455**, b=8 (FT right/OTS wrong), c=1.
- Multi-seed: **PENDING background training** (3 seeds fold 3).
- Point deltas: ΔTop-1 +4.15pp, ΔTop-5 +0.44pp, ΔMAP +2.42pp.

## Table 12 — gate activity (pooled 2504)
- Active (g=1, non-empty read): **2504 (100.0%)** — reader ALWAYS returns a read on ePillID.
- of which final Top-1 correct: **2366 (94.49%)**.
- Rescued (visual wrong -> MIRA right): **110 (4.4%)**; Hurt (visual right -> MIRA wrong): **6 (0.2%)**.
- Inactive (g=0, empty read): **0 (0.0%)**.
- INTEGRITY: on ePillID the gate never fires (imprints always readable); the low harm is due to imprint_match rarely outranking, NOT gate=0. Prop 1 stays as conditional theory; empirical wording must reflect this.
- Hurt-cause breakdown (6 cases): to inspect individually (hallucinated/misread vs shared-token correct read).

## Table 16 — reader ablation Top-5
- Fusion + off-the-shelf reader Top-5 = **99.84** (same as fine-tuned).

## Table 22 — stratified by imprint availability (pooled 2504)
- Imprint present, read non-empty: n=**2408**, visual Top-1 **90.49%**, ΔTop-1 **+4.53pp**.
- Imprint present, read empty: n=**0** (reader never empty on ePillID).
- Imprint absent in metadata: n=**96**, visual Top-1 **86.46%**, ΔTop-1 **-5.21pp** (fusion HURTS here — honest; motivates IDF/gate).

## Group B (to run): OCR baselines Table 5 (PaddleReader/PP-OCRv3 EXISTS in code; Tesseract needs pytesseract), soft-gate Table 24 (needs reader logprobs), Fig 8 qualitative images.
## Group C (background): multi-seed Table 11, modern backbone Table 23.

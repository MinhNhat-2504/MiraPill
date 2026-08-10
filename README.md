# MIRA-Pill

**M**ultimodal **I**mprint **R**eading **A**nd confidence-gated re-ranking for fine-grained pill
identification.

The hardest cases in pill identification are medications that look almost identical and differ only
in the code printed or debossed on the tablet. The ePillID benchmark named reliable imprint reading
as its most important open problem and left it unsolved, because the OCR available at the time was
not dependable on small, low-contrast pill surfaces.

This repository is an attempt at that problem with a modern vision–language model. It couples two
branches and lets them correct each other:

- a **visual branch** — ResNet50 with Compact Bilinear Pooling, trained with metric learning, which
  is strong at overall appearance but confuses near-identical pills;
- an **imprint branch** — Qwen2-VL-2B fine-tuned with LoRA to transcribe the imprint string, which
  is matched against a class-imprint database;
- a **confidence-gated, log-linear fusion** that re-ranks the top-*K* visual candidates:

```
S_final(q, c) = Ŝ_vis(q, c) + λ · g(q) · m(t_q, T_c)
```

where `t_q` is the string read from the query, `T_c` the database imprint of class `c`, `m(·)` a
token-overlap match, and `g(q)` a gate that switches the imprint term off when nothing was read.
The gate is what makes the method safe on pills that carry no imprint at all: when `g(q) = 0` the
ranking is exactly the visual ranking, so the imprint branch can help but cannot hurt.

The work is evaluated on three datasets spanning the full range of imprint availability — ePillID
(ground-truth imprints), CURE (imprints present but no metadata, so the database is built by OCR
consensus) and OGYEIv2 (essentially no imprints, used as a no-harm stress test).

> A paper describing this work is under review. Results are therefore not reproduced here; this
> repository contains the implementation only.

## Repository layout

```
src/
  finetune_qwen_imprint.py   LoRA fine-tuning of the Qwen2-VL imprint reader
  prep_imprint_data.py       builds the reader's fine-tuning split (reference images, by pill type)
  imprint_feasibility.py     reader evaluation; supports the qwen / paddle / tesseract readers
  fusion_rerank.py           caches imprint reads and applies the gated fusion
  fuse_4fold.py              aggregates the cross-validation rounds
  e2_significance.py         McNemar, Wilcoxon, bootstrap CIs
  e3_failure_analysis.py     rescued / hurt case analysis
  e4_openset.py              open-set recognition
  e5_ablation.py             branch ablation and λ / K sensitivity
  e6_clip_zeroshot.py        CLIP zero-shot baseline
  softgate_eval.py           soft confidence gate (log-probability calibrated)
  cure_*.py, ogyei_*.py      cross-dataset pipelines (cropping, imprint DB, fusion)
  rebuttal_stats*.py         clustered bootstrap and pooled statistics
  train_nocv.py, train_cv.py visual-branch training (from the ePillID benchmark)
  models/                    visual-branch model code, incl. vendored fast-MPN-COV
docker/                      conda environment used for the visual branch
```

## Getting started

### Data

None of the datasets are redistributed here. Get them from their original sources:

| Dataset | Source |
| --- | --- |
| ePillID | <https://github.com/usuyama/ePillID-benchmark> |
| CURE | <https://github.com/suiyiling/Few-shot-pill-recognition> |
| OGYEIv2 | <https://github.com/richardRadli/pill_detection> |
| Pillbox master data (imprint labels) | public NIH Pillbox release; see `fetch_pillbox_kaggle.py` |

Most scripts take the dataset roots as command-line arguments; the defaults point at the paths used
during development (`D:/Data/...`), so pass your own or edit the defaults.

### Environment

The two branches run under different environments, which is deliberate — the visual branch follows
the original ePillID release so the baseline stays comparable:

- **visual branch**: Python 3.6.7, PyTorch 1.3.1, torchvision 0.4.2, CUDA 10.1 (see
  `docker/conda/epillidpy36_env.yml`)
- **imprint branch**: Python 3.11, PyTorch 2.5.1 (CUDA 12.1), Transformers 4.46.2, PEFT 0.13.2,
  safetensors 0.7.0; optional `tesserocr` for the classical OCR baseline. An 8 GB GPU is enough
  (fp16 + gradient checkpointing).

One trap worth knowing about: across Transformers releases the Qwen2-VL language tower changed
module path, so LoRA adapter keys saved by a newer release do not match the names an older one
expects. When they do not match, PEFT attaches nothing, raises no error, and the adapter silently
behaves like the base model. `remap_adapter_keys.py` renames the keys between the two layouts.

On Windows the cuDNN algorithm search inside the Qwen2-VL vision tower overflows the default 1 MB
thread stack; `run_bigstack.py` and `run_ft_bigstack.py` run any module on a 128 MB stack instead.

### Typical run

```bash
# 1. reader fine-tuning data (reference images only, split by pill type)
python prep_imprint_data.py --gt_csv <Pillbox.csv> --data_root_dir <ePillID_root>

# 2. fine-tune the imprint reader
python run_ft_bigstack.py --out_dir qwen_imprint_lora

# 3. train the visual branch
python run_visual_train.py

# 4. how well does the reader read? (feasibility / before-after tables)
python imprint_feasibility.py --readers qwen --by_pill 2000 --adapter qwen_imprint_lora \
    --gt_csv <Pillbox.csv> --data_root_dir <ePillID_root>

# 5. cache the reads, then fuse and evaluate
python fusion_rerank.py --cache_imprints --imprint_cache fusion_out/imprints.csv
python fuse_4fold.py
python e2_significance.py && python e3_failure_analysis.py
```

Cross-dataset runs follow the same shape: build crops, build the OCR-consensus imprint database,
then fuse (`cure_build_crops.py` → `cure_imprint_db.py` → `cure_fusion.py`).

## Credits

The visual branch and the benchmark tooling come from the **ePillID benchmark** by Usuyama et al.
(MIT licensed), which this work builds on and deliberately keeps unchanged so that the baseline
remains comparable:

```bibtex
@inproceedings{usuyama2020epillid,
  title={ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification},
  author={Usuyama, Naoto and Delgado, Natalia Larios and Hall, Amanda K and Lundin, Jessica},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}
```

`src/models/fast-MPN-COV/` is vendored from [fast-MPN-COV](https://github.com/jiangtaoxie/fast-MPN-COV)
by Peihua Li and Jiangtao Xie (MIT). The imprint reader is
[Qwen2-VL-2B](https://github.com/QwenLM/Qwen2-VL); the OCR baselines are Tesseract and PP-OCRv3.

## License

MIT — see [LICENSE](LICENSE). The vendored components keep their own MIT licenses and copyright
notices.

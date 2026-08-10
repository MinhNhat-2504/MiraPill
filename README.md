# MIRA-Pill

**M**ultimodal **I**mprint **R**eading **A**nd confidence-gated re-ranking — nhận dạng thuốc viên
bằng cách đọc chữ dập trên viên và hợp nhất với đặc trưng thị giác.

Ca khó nhất của bài toán nhận dạng thuốc là những viên trông gần như giống hệt nhau, chỉ khác nhau ở
dãy ký tự được in hoặc dập chìm trên bề mặt. Benchmark ePillID từng chỉ đích danh việc đọc tin cậy
dãy ký tự này là bài toán mở quan trọng nhất của họ, và để ngỏ nó — vì OCR ở thời điểm đó không đủ
tin cậy trên bề mặt viên thuốc nhỏ, tương phản thấp.

Repo này là một hướng giải quyết bài toán đó bằng vision–language model hiện đại. Hệ thống gồm hai
nhánh bổ khuyết cho nhau:

- **Nhánh thị giác** — ResNet50 kết hợp Compact Bilinear Pooling, huấn luyện theo metric learning.
  Nhánh này mạnh ở tổng thể hình dáng, màu sắc, nhưng hay nhầm giữa các viên gần giống nhau.
- **Nhánh imprint** — Qwen2-VL-2B fine-tune bằng LoRA để đọc dãy ký tự trên viên, rồi so khớp chuỗi
  đọc được với cơ sở dữ liệu imprint của từng lớp.
- **Hợp nhất log-tuyến tính có cổng tin cậy**, xếp hạng lại top-*K* ứng viên tốt nhất của nhánh thị
  giác:

```
S_final(q, c) = Ŝ_vis(q, c) + λ · g(q) · m(t_q, T_c)
```

trong đó `t_q` là chuỗi đọc được từ ảnh truy vấn, `T_c` là imprint của lớp `c` trong cơ sở dữ liệu,
`m(·)` là độ khớp token, còn `g(q)` là cổng tin cậy — tắt hẳn số hạng imprint khi không đọc được gì.

Chính cái cổng này giúp phương pháp an toàn với những viên không hề có imprint: khi `g(q) = 0`, thứ
tự xếp hạng đúng bằng thứ tự của nhánh thị giác, nên nhánh imprint có thể giúp thêm chứ không thể
làm hỏng kết quả.

Hệ thống được đánh giá trên ba bộ dữ liệu trải đủ phổ imprint — ePillID (có nhãn imprint chuẩn), CURE
(viên có imprint nhưng không có metadata, nên cơ sở dữ liệu imprint được dựng bằng OCR đồng thuận) và
OGYEIv2 (gần như không có imprint, dùng làm phép thử "không gây hại").

> Bài báo mô tả công trình này đang trong quá trình phản biện, nên repo chỉ chứa phần cài đặt, không
> kèm kết quả thực nghiệm.

## Cấu trúc thư mục

```
src/
  finetune_qwen_imprint.py   fine-tune LoRA cho reader Qwen2-VL
  prep_imprint_data.py       dựng tập fine-tune cho reader (ảnh reference, chia theo loại thuốc)
  imprint_feasibility.py     đánh giá reader; hỗ trợ ba engine qwen / paddle / tesseract
  fusion_rerank.py           cache chuỗi imprint đọc được và áp dụng fusion có cổng
  fuse_4fold.py              tổng hợp kết quả các vòng cross-validation
  e2_significance.py         kiểm định McNemar, Wilcoxon, khoảng tin cậy bootstrap
  e3_failure_analysis.py     phân tích ca được cứu / bị hỏng sau khi fusion
  e4_openset.py              nhận dạng open-set
  e5_ablation.py             ablation từng nhánh và độ nhạy theo λ / K
  e6_clip_zeroshot.py        baseline CLIP zero-shot
  softgate_eval.py           cổng tin cậy mềm, hiệu chỉnh theo log-probability
  cure_*.py, ogyei_*.py      pipeline cho hai bộ dữ liệu chéo (cắt ảnh, dựng DB imprint, fusion)
  rebuttal_stats*.py         bootstrap phân cụm và thống kê gộp
  train_nocv.py, train_cv.py huấn luyện nhánh thị giác (kế thừa từ ePillID benchmark)
  models/                    mã nguồn nhánh thị giác, gồm cả fast-MPN-COV đi kèm
docker/                      môi trường conda dùng cho nhánh thị giác
```

## Bắt đầu

### Dữ liệu

Repo không phát hành lại bất kỳ bộ dữ liệu nào. Bạn tải trực tiếp từ nguồn gốc:

| Bộ dữ liệu | Nguồn |
| --- | --- |
| ePillID | <https://github.com/usuyama/ePillID-benchmark> |
| CURE | <https://github.com/suiyiling/Few-shot-pill-recognition> |
| OGYEIv2 | <https://github.com/richardRadli/pill_detection> |
| Pillbox master data (nhãn imprint) | bản phát hành công khai của NIH Pillbox; xem `fetch_pillbox_kaggle.py` |

Hầu hết script nhận đường dẫn dữ liệu qua tham số dòng lệnh. Giá trị mặc định đang trỏ tới đường dẫn
lúc phát triển (`D:/Data/...`), nên bạn truyền đường dẫn của mình vào hoặc sửa lại mặc định.

### Môi trường

Hai nhánh chạy trên hai môi trường khác nhau, và điều này là cố ý: nhánh thị giác giữ đúng môi trường
của bản phát hành ePillID gốc để baseline vẫn so sánh được.

- **Nhánh thị giác**: Python 3.6.7, PyTorch 1.3.1, torchvision 0.4.2, CUDA 10.1 (xem
  `docker/conda/epillidpy36_env.yml`).
- **Nhánh imprint**: Python 3.11, PyTorch 2.5.1 (CUDA 12.1), Transformers 4.46.2, PEFT 0.13.2,
  safetensors 0.7.0; thêm `tesserocr` nếu cần chạy baseline OCR cổ điển. GPU 8 GB là đủ (fp16 kèm
  gradient checkpointing).

Có một cái bẫy nên biết trước: qua các phiên bản Transformers, đường dẫn module của language tower
trong Qwen2-VL bị đổi, nên key của adapter LoRA lưu bởi bản mới sẽ không khớp tên mà bản cũ mong đợi.
Khi tên không khớp, **PEFT không gắn trọng số nào cả, cũng không báo lỗi**, và adapter im lặng hoạt
động y như model gốc chưa fine-tune. Script `remap_adapter_keys.py` dùng để đổi tên key qua lại giữa
hai kiểu.

Trên Windows còn một vấn đề nữa: quá trình dò thuật toán cuDNN bên trong vision tower của Qwen2-VL
làm tràn stack 1 MB mặc định của thread. Hai script `run_bigstack.py` và `run_ft_bigstack.py` chạy
module bất kỳ trên thread có stack 128 MB để tránh lỗi này.

### Chạy thử

```bash
# 1. dựng dữ liệu fine-tune cho reader (chỉ ảnh reference, chia theo loại thuốc)
python prep_imprint_data.py --gt_csv <Pillbox.csv> --data_root_dir <ePillID_root>

# 2. fine-tune reader imprint
python run_ft_bigstack.py --out_dir qwen_imprint_lora

# 3. huấn luyện nhánh thị giác
python run_visual_train.py

# 4. reader đọc tốt tới đâu? (bảng khả thi / trước - sau fine-tune)
python imprint_feasibility.py --readers qwen --by_pill 2000 --adapter qwen_imprint_lora \
    --gt_csv <Pillbox.csv> --data_root_dir <ePillID_root>

# 5. cache chuỗi đọc được, rồi fusion và đánh giá
python fusion_rerank.py --cache_imprints --imprint_cache fusion_out/imprints.csv
python fuse_4fold.py
python e2_significance.py && python e3_failure_analysis.py
```

Hai bộ dữ liệu chéo chạy theo trình tự tương tự: cắt ảnh viên thuốc, dựng cơ sở dữ liệu imprint bằng
OCR đồng thuận, rồi fusion (`cure_build_crops.py` → `cure_imprint_db.py` → `cure_fusion.py`).

## Ghi nhận

Nhánh thị giác và phần khung đánh giá kế thừa từ **ePillID benchmark** của Usuyama và cộng sự (giấy
phép MIT). Công trình này giữ nguyên phần đó, không chỉnh sửa, để baseline vẫn so sánh được với các
kết quả đã công bố trên benchmark:

```bibtex
@inproceedings{usuyama2020epillid,
  title={ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification},
  author={Usuyama, Naoto and Delgado, Natalia Larios and Hall, Amanda K and Lundin, Jessica},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}
```

Thư mục `src/models/fast-MPN-COV/` lấy từ [fast-MPN-COV](https://github.com/jiangtaoxie/fast-MPN-COV)
của Peihua Li và Jiangtao Xie (MIT). Reader imprint dùng
[Qwen2-VL-2B](https://github.com/QwenLM/Qwen2-VL); hai baseline OCR là Tesseract và PP-OCRv3.

## Giấy phép

MIT — xem [LICENSE](LICENSE). Các thành phần đi kèm giữ nguyên giấy phép MIT và thông tin bản quyền
của tác giả gốc.

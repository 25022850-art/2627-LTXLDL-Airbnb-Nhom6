# Bài tập lớn cuối kỳ — Inside Airbnb tại Porto

**Môn học:** Lập trình xử lý dữ liệu (UET.DSE2049), học kỳ 1, năm học 2026–2027.

Dự án phân tích thị trường Airbnb tại Porto qua bốn mốc dữ liệu. Quy trình làm sạch dữ liệu chỗ ở, lịch và đánh giá, gán nhãn đánh giá bằng baseline và Google Gemini, đối chiếu với nhãn tay, tính KPI và xuất biểu đồ, bản đồ. [Đề bài chính thức](https://courses.iaidev.com/programming-for-data-processing/2627-1/projects/project_airbnb.html)

## Thông tin nhóm

- **Mã lớp:** UET.DSE2049 3
- **Tên/mã nhóm:** Nhóm 6
- **Thành phố được phân công:** Porto, Bồ Đào Nha
- **Bên đặt hàng phân tích:**
- **Bốn mốc dữ liệu:** `2025-09-21`, `2025-12-25`, `2026-03-26`, `2026-06-23`

| Họ tên | MSSV | GitHub username | Công việc phụ trách |
|---|---|---|---|
| Lê Chí Hiếu | 25022850 | 25022850-art | Kpi, llm, viz, download, baseline |
| Nguyễn Thanh Hải | 25022804 | HAITO192 | Qa, download, llm, geojson, baseline, evaluate  |
| Lê Hồng Đăng | 25022823 | HNDCMMRCT | |
| Hồ Thành Đạt | 25022818 | 25022818-HoThanhDat	| Tạo biểu đồ Viz |
| Đinh Văn Lượng | 25022928 | 25022928-lab | |

## Cài đặt và chạy

### Môi trường

Sử dụng Python 3.14.4 theo [`.python-version`](.python-version). Chạy các lệnh dưới đây từ thư mục gốc của repo. Ví dụ dùng Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Phiên bản các thư viện được ghim trong [`requirements.txt`](requirements.txt).

### Cấu hình và chuẩn bị dữ liệu

Chỉnh [`configs/porto.yaml`](configs/porto.yaml) để thay đổi các mốc trong `snapshots`, quy tắc làm sạch trong `qa`, tham số trong `kpi` và mô hình, kích thước mẫu, đường dẫn cache trong `llm`. Cấu hình hiện dùng EUR và lấy mốc `2026-06-23` cho các KPI theo mốc đơn. Trình tải lọc theo Lớp 3, Nhóm 6 từ hai bảng danh sách ID trong cấu hình. Một số bước vẫn tham chiếu trực tiếp đến cấu hình Porto.

Trên bản clone mới, tải và lọc dữ liệu Inside Airbnb trước. Lệnh này cần kết nối đến Inside Airbnb và hai bảng Google Sheets trong cấu hình.

```powershell
.\.venv\Scripts\python.exe -m src.download
```

Mỗi thư mục `data/raw/<mốc>/` chứa `listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz` và `neighbourhoods.geojson`. [Nguồn dữ liệu Inside Airbnb](https://insideairbnb.com/get-the-data/)

### Chạy toàn bộ pipeline

Đặt khóa Google Gemini API qua biến môi trường `GEMINI_API_KEY` hoặc trong tệp `.env` ở thư mục gốc với dòng `GEMINI_API_KEY=<khóa của bạn>`. Mô hình được đọc từ `llm.provider`. Bản hiện tại khởi tạo Gemini client trước khi tra cache nên cần khóa ngay cả khi các đánh giá đã có nhãn trong cache.

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --config configs/porto.yaml
```

Lệnh thực hiện QA, gán nhãn baseline và LLM, đánh giá bằng nhãn tay, tính KPI và vẽ hình. Có thể thêm `--skip-llm` để dùng lại `data/processed/llm_output.csv` hiện có cho các bước phía sau.

### Tệp kết quả để đối chiếu

| Kết quả | Vị trí |
|---|---|
| Dữ liệu đã làm sạch | `data/processed/<mốc>/processed_{listings,calendar,reviews}.csv` |
| Báo cáo QA | `reports/<mốc>/qa_report_{listings,calendar,reviews}.csv` |
| KPI | `data/processed/kpi_*.csv` |
| Nhãn baseline và LLM | `data/processed/baseline_output.csv`, `data/processed/llm_output.csv` |
| Cache LLM | `data/processed/llm_cache.jsonl` |
| Đánh giá và kiểm tra nhãn LLM | `reports/f1_score.csv`, `reports/audit_llm.csv`, `reports/audit_llm_summary.csv` |
| Biểu đồ và bản đồ | [`figures/`](figures/), gồm biểu đồ KPI và bản đồ RevPAR, phân bố nhãn theo khu vực |
| Báo cáo tổng hợp | |

## Lưu ý khi làm bài

- `.gitignore` loại trừ dữ liệu thô trong `data/raw/`, môi trường `.venv/` và tệp `.env`.
- Ghi nhận việc sử dụng AI trong [`AI_USAGE.md`](AI_USAGE.md).
- Đối chiếu yêu cầu nộp bài và lịch vấn đáp trên Canvas và trong đề bài chính thức.

## Nộp bài

Sau khi đẩy commit cuối lên GitHub, tạo và đẩy tag `final` theo đề bài. Ghi SHA của commit nộp vào báo cáo.

```bash
git tag final
git push origin final
```

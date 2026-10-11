## 1. Yêu cầu môi trường

- **Python**: Phiên bản ghi trong tệp [`.python-version`] [`.python-version`](.python-version) (3.14.4)
- Hệ điều hành đã chạy thử : `<Windows 11>`.

## 2. Cài đặt 
```bash
# 1. Clone repo
git clone https://github.com/25022850-art/2627-LTXLDL-Airbnb-Nhom6.git
cd 2627-LTXLDL-Airbnb-Nhom6

# 2. Tạo môi trường ảo
python -m venv .venv

# Linux / macOS
#source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Cài thư viện 
pip install -r requirements.txt
```

## 3. Cách chạy / chấm lại 
*Chạy toàn bộ quy trình bằng một lệnh:*

```bash
python run_pipeline.py --config configs/<city>.yaml
```
*Các tuỳ chọn hữu ích:*
```bash
# Bỏ qua bước LLM 
python run_pipeline.py --config configs/<city>.yaml --skip-llm

# Chạy nhanh trên mẫu nhỏ trong data/samples/
python run_pipeline.py --config configs/<city>.yaml --sample
```

*Chạy riêng từng bước:*

python -m src.download  --config configs/<city>.yaml
python -m src.qa        --config configs/<city>.yaml
python -m src.kpi       --config configs/<city>.yaml
python -m src.llm       --config configs/<city>.yaml
python -m src.evaluate  --config configs/<city>.yaml
python -m src.viz       --config configs/<city>.yaml
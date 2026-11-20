import argparse
from pathlib import Path
import yaml

from src.download import download_raw
from src.exec_qa import exec_qa
from src.exec_baseline import exec_baseline
from src.exec_llm import exec_llm
from src.exec_evaluate import exec_eval
from src.exec_kpi import exec_kpi


# PROJECT_ROOT = Path(__file__).resolve().parent
# CONFIG_PATH = PROJECT_ROOT / "configs" / "porto.yaml"


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua LLM")
    parser.add_argument("--config", required=True, type=str, help="Chọn file config thành phố mong muốn")
    args = parser.parse_args()

    skip_llm = args.skip_llm

    config_path = args.config
    config = load_config(config_path)

    # print("Bước: Tải dữ liệu raw")
    # download_raw(CONFIG_PATH)
    # print("Tải dữ liệu raw hoàn tất")

    print("Bước: Làm sạch dữ liệu (QA)")
    exec_qa(config)
    print("Hoàn thành bước làm sạch")

    print("Bước: Gán nhãn comments bằng baseline")
    exec_baseline(config, config_path)
    print("Hoàn thành gán nhãn bằng baseline")

    if not skip_llm:
        print("Bước: Gọi LLM gán nhãn comments")
        exec_llm(config, config_path)
        print("Hoàn thành gán nhãn bằng LLM")

    print("Bước: Đánh giá baseline và llm")
    exec_eval()
    print("Hoàn thành việc đánh giá baseline và llm")

    print("Bước: Tính toán KPI")
    exec_kpi()
    print("Hoàn thành việc tính KPI")
    
    print("\nHoàn tất pipeline.")


if __name__ == "__main__":
    run_pipeline()
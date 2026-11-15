from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

import os
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from google import genai
import json
import yaml
import time

load_dotenv()

class ReviewInfo(BaseModel):
    general_sentiment: Literal["Positive", "Mixed", "Negative"]
    accessibility: Literal["Highly accessible", "Poor access", "Neutral", "Not mentioned"]
    noise: Literal["Quiet", "Noisy", "Acceptable", "Not mentioned"]
    
PROMPT_TEMPLATE = """Bạn là một chuyên gia phân tích dữ liệu đánh giá chỗ ở. Nhiệm vụ của bạn là trích xuất thông tin và phân loại cảm xúc theo đúng định dạng JSON.

### ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (Chỉ trả về JSON, không giải thích thêm):
{{
  "general_sentiment": "Positive" | "Negative" | "Mixed",
  "accessibility": "Highly accessible" | "Poor access" | "Neutral" | "Not mentioned",
  "noise": "Noisy" | "Acceptable" | "Quiet" | "Not mentioned"
}}

### QUY TẮC GÁN NHÃN CHI TIẾT:

1. general_sentiment (Cảm xúc tổng thể):
- "Mixed": Gán khi có cả khen và chê trong cùng một đánh giá, số lượng chê đáng kể hoặc khi đánh giá không rõ ràng, mâu thuẫn.
- "Positive": CHỈ gán khi đánh giá hoàn toàn là khen ngợi hoặc khen ngợi chiếm ưu thế rõ ràng.
- Ngoại lệ: Đánh giá trống/không có nội dung chữ -> gán "Mixed".

2. accessibility (Vị trí & Đi lại):
- "Neutral" ( ƯU TIÊN KIỂM TRA TRƯỚC, CHỈ LÀ KIỂM TRA TRƯỚC, KHÔNG GÁN NEUTRAL MÀ KHÔNG SUY NGHĨ KĨ): BẮT BUỘC gán nhãn này nếu đánh giá dùng các cụm từ thể hiện mức độ vừa phải như: "central enough", "location is ok", "good enough", "fine", "not bad".
- "Highly accessible": CHỈ gán khi có thông tin CỤ THỂ về sự thuận tiện đi lại (khoảng cách, thời gian đi bộ, gần bến xe/ga tàu/tiện ích) HOẶC dùng từ "central/center/centre" kết hợp với location (chỉ khi không có từ enough hoặc các tính từ trong Neutral). Gán khi review nhấn mạnh khoảng cách gần như "near to", "walking distance", "close to" các địa điểm.
- "Not mentioned": BẮT BUỘC gán nhãn này nếu khách chỉ khen vị trí chung chung bằng tính từ mạnh (ví dụ: "perfect location", "wonderful location", "great location") mà KHÔNG nhắc đến khoảng cách đi lại. Không tự suy diễn. Có thể gán nhãn Highly accessible nếu sau tính từ mạnh có thông tin về khoảng cách tới các tiện ích ("ví dụ :perfect location to visit the museum" -> Highly accessible).Nếu chỉ nói chung chung là "perfect location to visit Porto" thì vẫn là Neutral, chỉ khi là một tiện ích cụ thể (ví dụ: "museum", "park", "restaurant", "store") mới gán Highly accessible.


3. noise (Âm thanh & Tiếng ồn):
- "Quiet": ( ƯU TIÊN KIỂM TRA TRƯỚC, CHỈ LÀ KIỂM TRA TRƯỚC, KHÔNG GÁN QUIET MÀ KHÔNG SUY NGHĨ KĨ ) CHỈ gán khi khách nhấn mạnh sự yên tĩnh về mặt âm thanh (ví dụ: "quiet", "no noise at all"). "Quite" có thể hiểu là viết sai chính tả cho quiet và gán nhãn "Quiet".
- "Not mentioned": BẮT BUỘC gán nhãn này với các từ chỉ không gian/bầu không khí chung chung như "peaceful", "calm", "relaxing" (chúng mô tả cảm giác thư thái, KHÔNG ĐỦ để kết luận về âm thanh). Không tự suy diễn khi không nhắc trực tiếp đến âm thanh.
- "Acceptable": Gán khi có tiếng ồn nhưng ở mức chịu đựng được ("not too loud", "acceptable noise", "little noisy").
- "Noisy": Chỉ gán khi tiếng ồn lớn, ảnh hưởng tiêu cực mạnh đến trải nghiệm.

Review: {text}"""

DINH_DANG = {"type": "text", "mime_type": "application/json",
             "schema": ReviewInfo.model_json_schema()}

def get_api_key():
    try:
        return os.environ.get("GEMINI_API_KEY")
    except Exception:                     
        raise Exception("Hãy tạo file .env ở folder của dự án và gán API KEY vào biến 'GEMINI_API_KEY'")

def get_model(config):
    try:
        MODEL = config.get("llm").get("provider")
        return MODEL
    except Exception:
        raise Exception(f"Có lỗi khi lấy 'provider' ở mục 'llm' từ file config; MODEL={MODEL}")

def create_client(config):
    API_KEY = get_api_key()
    CO_KHOA = bool(API_KEY)
    if CO_KHOA:
        client = genai.Client(api_key=API_KEY)
        return client
    else:
        raise Exception(f"Có lỗi xảy ra ở bước create_client, CO_KHOA={CO_KHOA}, hãy kiểm tra lại API KEY")
    
def get_cache_path(config):
    try:
        cache_path = Path(config.get("llm").get("cache"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    except Exception:
        raise Exception(f"Có lỗi ở bước get_cache_path, cache_path={cache_path}")

def get_llm_cache(cache_path):
    try:
        if cache_path.exists():
            return pd.read_json(cache_path, lines=True) 
        else : 
            return pd.DataFrame(columns=[
                "review_text", 
                "general_sentiment", 
                "accessibility", 
                "noise"
            ])
    except Exception as e:
        raise Exception(f"Có lỗi khi đọc file cache tại {cache_path}: {e}")

def call_llm(review, model, client):
    try : 
        prompt = PROMPT_TEMPLATE.format(text=review)
        tra_loi = client.interactions.create(
            model=model,
            input=prompt,
            response_format=DINH_DANG
        )

        result_text = tra_loi.output_text
        info = ReviewInfo.model_validate_json(result_text)
        return info.model_dump()
    except Exception as e:
        raise Exception(f"Lỗi khi gọi LLM hoặc parse JSON: {e}")

def label(cache_path, cache_llm, rv, client, model):
    if not cache_llm.empty and 'comments' in cache_llm.columns:
        matched_row = cache_llm[cache_llm['comments'] == rv]
        if not matched_row.empty:
            result_dict = matched_row.iloc[0].to_dict() #chi lay 1 dong
            return result_dict
        
    result_dict = call_llm(rv, model, client)
    result_dict['comments'] = rv

    new_row_df = pd.DataFrame([result_dict])
    new_row_df.to_json(
        cache_path, 
        orient='records', # dinh dang la JSONL
        lines=True, 
        mode='a', 
    )   
    print(f"Luu thanh cong ra file {cache_path}")
    time.sleep(4)
    return result_dict 

def llm(rv, config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model = get_model(config)
    client = create_client(config)
    cache_path = get_cache_path(config)
    cache_llm = get_llm_cache(cache_path)

    rv = rv.copy()
    result = rv["comments"].apply(lambda comment: label(cache_path, cache_llm, comment, client, model))
    result = result.apply(pd.Series)

    return result

def audit_llm_predictions(df):
    flags = []
    for idx, row in df.iterrows():
        text = str(row["comments"]).strip()
        word_count = len(text.split())
        gen, acc, noise = row["general_sentiment"], row["accessibility"], row["noise"]
        row_flags = []

        # 1. Rỗng nhưng vẫn gán nhãn khía cạnh
        if word_count == 0 and (acc != "Not mentioned" or noise != "Not mentioned"):
            row_flags.append("EMPTY_TEXT_WITH_ASPECT")

        # 2. Mâu thuẫn cực tính chéo
        if gen == "Positive" and ("Negative" in [acc, noise]):
            row_flags.append("CONFLICT_POS_GEN_WITH_NEG_ASPECT")
        if gen == "Negative" and ("Positive" in [acc, noise]):
            row_flags.append("CONFLICT_NEG_GEN_WITH_POS_ASPECT")
        if {acc, noise} == {"Positive", "Negative"} and gen != "Mixed":
            row_flags.append("MISSING_MIXED_SENTIMENT")

        # 3. Câu ngắn (<= 5 từ) nhưng bắt cả 2 khía cạnh
        mentioned_count = sum(x != "Not mentioned" for x in [acc, noise])
        if 0 < word_count <= 5 and mentioned_count > 1:
            row_flags.append("SHORT_TEXT_MULTI_ASPECT")

        flags.append(", ".join(row_flags) if row_flags else "OK")
    
    df["audit_flag"] = flags
    return df
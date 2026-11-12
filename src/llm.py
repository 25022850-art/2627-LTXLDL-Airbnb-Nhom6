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

load_dotenv()

class ReviewInfo(BaseModel):
    general_sentiment: Literal["Positive", "Mixed", "Negative"]
    accessibility: Literal["Highly accessible", "Poor access", "Neutral", "Not mentioned"]
    noise: Literal["Quiet", "Noisy", "Acceptable", "Not mentioned"]
    
PROMPT_TEMPLATE = (
    "Bạn là công cụ trích xuất thông tin từ đánh giá về chỗ ở.\n"
    "Phân tích đánh giá sau và điền đúng schema. "
    "Chỉ gán khía cạnh THỰC SỰ được nhắc đến;\n"
    'đánh giá không có nội dung thì trả sentiment "mixed", '
    'accessibility "Not mentioned" và noise "Not mentioned" '
    "và các danh sách rỗng.\n\n"
    "Hãy chỉ trả về duy nhất một object JSON hợp lệ chứa 3 key: "
    "'general_sentiment', 'accessibility', và 'noise'. "
    "Không kèm theo bất kỳ văn bản giải thích nào khác.\n\n"
    "Review: {text}"
)
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
        # result_text = result_text.replace("```json", "").replace("```", "").strip()
        info = ReviewInfo.model_validate_json(result_text)
        return info.model_dump()
    except Exception as e:
        raise Exception(f"Lỗi khi gọi LLM hoặc parse JSON: {e}")
        

def llm(rv, config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    cache_path = get_cache_path(config)
    model = get_model(config)
    client = create_client(config)

    cache_llm = get_llm_cache(cache_path)
    if not cache_llm.empty and 'review_text' in cache_llm.columns:
        matched_row = cache_llm[cache_llm['review_text'] == rv]
        if not matched_row.empty:
            result_dict = matched_row.iloc[0].to_dict() #chi lay 1 dong
            result_dict.pop('review_text', None)
            return result_dict
        
    result_dict = call_llm(rv, client, model)
    cache_record = result_dict.copy()
    cache_record['review_text'] = rv

    new_row_df = pd.DataFrame([cache_record])
    new_row_df.to_json(
        cache_path, 
        orient='records', # dinh dang la JSONL
        lines=True, 
        mode='a', 
    )

    return result_dict

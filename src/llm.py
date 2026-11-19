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

### QUY TẮC GÁN NHÃN CHI TIẾT (Áp dụng nghiêm ngặt theo từng bước):

1. general_sentiment (Cảm xúc tổng thể):
- "Positive": Gán khi đánh giá hoàn toàn là khen ngợi, HOẶC lời khen áp đảo hoàn toàn lời chê. Lời chê (nếu có) chỉ là những bất tiện rất nhỏ, lướt qua, không ảnh hưởng đến chất lượng phòng ốc (ví dụ: "trời hôm đó mưa", "hơi khó tìm hộp đựng chìa khóa lúc đầu", "thiếu một cuộn giấy vệ sinh").
- "Mixed": Gán khi lời chê và khen gần bằng nhau, HOẶC khi đánh giá có nhiều lời khen nhưng đi kèm với những phàn nàn ĐÁNG KỂ về cơ sở vật chất/không gian sống (ví dụ: có mùi hôi/funky smell, phòng quá nhỏ/cramped, không có chỗ để hành lý, quá lạnh/cold, đồ đạc cũ hỏng/small holes). 
  -> LƯU Ý QUAN TRỌNG: Kể cả khi khách dùng cấu trúc "khen + chê + chốt lại bằng lời xoa dịu" (ví dụ: "The charm outweighs the negatives", "Apart from that it was great", "Everything is OK", "We would stay here again"), thì sự xuất hiện của các điểm trừ vật lý rõ ràng này vẫn BẮT BUỘC phải gán nhãn "Mixed".
- "Negative": Gán khi đánh giá chủ yếu là phàn nàn, chê bai, hoặc sự thất vọng/tức giận chiếm ưu thế rõ ràng.
- Ngoại lệ: Đánh giá trống, chỉ có ký tự đặc biệt hoặc không có nội dung chữ -> BẮT BUỘC gán "Mixed".

2. accessibility (Vị trí & Khả năng đi lại):
- BƯỚC 1 - KIỂM TRA SỰ LIÊN QUAN: Đọc xem đánh giá có nhắc trực tiếp đến VỊ TRÍ (location, area, neighborhood) hoặc VIỆC DI CHUYỂN/KHOẢNG CÁCH (walking, bus, metro, near, close, far...) hay không.
  -> Nếu KHÔNG nhắc đến vị trí/đi lại: BẮT BUỘC gán "Not mentioned" và dừng xét mục này.
  -> Nếu chỉ khen vị trí chung chung bằng tính từ mạnh ("perfect location", "great location", "wonderful area") mà KHÔNG nói rõ gần tiện ích/điểm tham quan cụ thể nào (như museum, park, station, restaurant...): BẮT BUỘC gán "Not mentioned". (Lưu ý: nếu nói chung chung "perfect location to visit Porto" vẫn gán "Not mentioned").
- BƯỚC 2 - PHÂN LOẠI KHI CÓ NHẮC ĐẾN VỊ TRÍ/ĐI LẠI (Xét theo thứ tự ưu tiên sau):
  + "Neutral": BẮT BUỘC gán khi khách đánh giá vị trí/đi lại ở mức trung bình, tạm ổn (ví dụ: "central enough", "location is ok", "location was fine", "area is not bad"). CHÚ Ý: Các từ "fine", "ok", "not bad" PHẢI đi kèm với chủ thể vị trí/khu vực mới được tính; nếu dùng cho phòng ở/chủ nhà ("room was fine") thì gán "Not mentioned".
  + "Highly accessible": Gán khi có thông tin CỤ THỂ về sự thuận tiện đi lại ("walking distance", "near to", "close to", gần bến xe/ga tàu/tiện ích cụ thể như museum, store) HOẶC dùng từ "central/center/centre" mô tả vị trí (mà không kèm từ giảm nhẹ như "enough").
  + "Poor access": Gán khi khách phàn nàn vị trí xa trung tâm, khó tìm đường, bất tiện đi lại hoặc leo dốc/cầu thang quá vất vả.

3. noise (Âm thanh & Tiếng ồn):
- BƯỚC 1 - KIỂM TRA SỰ LIÊN QUAN: Đọc xem đánh giá có từ ngữ nào mô tả TRỰC TIẾP về ÂM THANH, TIẾNG ĐỘNG hoặc SỰ YÊN TĨNH về thính giác hay không.
  -> Nếu KHÔNG nhắc trực tiếp đến âm thanh/tiếng ồn: BẮT BUỘC gán "Not mentioned" và dừng xét mục này.
  -> BẮT BUỘC gán "Not mentioned" với các từ chỉ bầu không khí/cảm giác chung chung như "peaceful", "calm", "relaxing" (không đủ cơ sở kết luận về tiếng ồn).
  -> CẢNH BÁO TỪ "QUITE": Từ "quite" (khá là, ví dụ: "quite clean", "quite nice") là phó từ chỉ mức độ, KHÔNG PHẢI lỗi chính tả của "quiet". Chỉ coi "quite" là "Quiet" khi ngữ cảnh rõ ràng đang nói về âm thanh/sự yên tĩnh (ví dụ: "street was very quite at night").
- BƯỚC 2 - PHÂN LOẠI KHI CÓ NHẮC ĐẾN ÂM THANH (Xét theo thứ tự ưu tiên sau):
  + "Quiet": CHỈ gán khi khách khen ngợi rõ ràng về sự yên tĩnh âm thanh ("quiet", "no noise at all", "soundproof", "silent at night").
  + "Acceptable": Gán khi có nhắc đến tiếng ồn nhưng ở mức chấp nhận được ("not too loud", "a little noisy", "acceptable noise").
  + "Noisy": Gán khi phàn nàn tiếng ồn lớn, gây khó chịu hoặc ảnh hưởng giấc ngủ.

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
    text = df["comments"].astype(str).str.strip()
    word_count = text.str.split().str.len()

    gen = df["general_sentiment"]
    acc = df["accessibility"]
    noise = df["noise"]

    acc_mentioned = acc != "Not mentioned"
    noise_mentioned = noise != "Not mentioned"
    mentioned_count = acc_mentioned.astype(int) + noise_mentioned.astype(int)

    rules = {
        # 1. Rỗng nhưng vẫn gán nhãn khía cạnh
        "EMPTY_TEXT_WITH_ASPECT": (word_count == 0) & (acc_mentioned | noise_mentioned),

        # 2. Mâu thuẫn cực tính chéo
        "CONFLICT_POS_GEN_WITH_NEG_ASPECT": (gen == "Positive") & ((acc == "Negative") | (noise == "Negative")),
        "CONFLICT_NEG_GEN_WITH_POS_ASPECT": (gen == "Negative") & ((acc == "Positive") | (noise == "Positive")),
        "MISSING_MIXED_SENTIMENT": (
            ((acc == "Positive") & (noise == "Negative")) |
            ((acc == "Negative") & (noise == "Positive"))
        ) & (gen != "Mixed"),

        # 3. Câu ngắn (<= 5 từ) nhưng bắt cả 2 khía cạnh
        "SHORT_TEXT_MULTI_ASPECT": (word_count > 0) & (word_count <= 5) & (mentioned_count > 1),
    }

    # Ghép các cờ: với mỗi dòng, nối tên các rule đang True bằng ", "
    flag_df = pd.DataFrame({name: np.where(mask, name, "") for name, mask in rules.items()}, index=df.index)
    combined = flag_df.agg(
        lambda r: ", ".join(v for v in r if v), axis=1
    )

    df["audit_flag"] = combined.replace("", "OK")
    return df
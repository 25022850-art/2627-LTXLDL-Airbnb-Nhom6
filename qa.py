from math import inf
from pathlib import Path

import pandas as pd
import re


ROOM_TYPES = {'Entire home/apt', 'Private room'}
AREAS = {'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08'}


def clean_text(series: pd.Series) -> pd.Series:
    text = series.str.strip()
    return text


def _flag_field(flags, before, field, valid):
    # Xem cột gốc để phân biệt ô trống với ô có giá trị sai như "abc".
    missing = before[field].isna()
    flags[f'{field}_missing'] = missing
    flags[f'{field}_invalid'] = ~missing & ~valid


def id_flags(data, prefix):
    """Tạo một cột True/False cho mỗi quy tắc kiểm tra ID."""
    ids = data['id']
    valid_id = ids.notna()
    repeated_id = valid_id & ids.duplicated()
    distinct_rows = data.loc[valid_id].drop_duplicates()
    conflicts = distinct_rows['id'].duplicated(keep=False)
    conflict_ids = distinct_rows.loc[conflicts, 'id']
    return pd.DataFrame({
        f'{prefix}_id_invalid': ~valid_id,
        f'{prefix}_duplicate_within': repeated_id,
        f'{prefix}_duplicate_conflict': ids.isin(conflict_ids),
    })


def _qa_numeric_field(before, after, flags, field, low, high, integer_only):
    """Kiểm tra một cột số: ghi cờ lỗi và tạo cột <field>_processed."""
    num = before[field]
    valid = (num >= low) & (num < inf)
    if high is not None:
        valid = valid & (num <= high)
    if integer_only:
        valid = valid & (num % 1 == 0)
    _flag_field(flags, before, field, valid)
    new_col = f'{field}_processed'
    after[new_col] = after[field]
    # after.loc[~valid, new_col] = None


def _qa_category_field(before, after, flags, field, allowed):
    """Kiểm tra một cột phân loại theo tập giá trị cho phép."""
    category = before[field]
    allowed_category = category.isin(allowed)
    _flag_field(flags, before, field, allowed_category)

    new_col = f'{field}_processed'
    after[new_col] = after[field]


def qa_host_id(before, flags):
    flags["host_id_missing"] = before["host_id"].isna()


def qa_number_of_reviews_ltm(before, after, flags):
    _qa_numeric_field(before, after, flags, 'number_of_reviews_ltm', 0, None, True)


def qa_number_of_reviews(before, after, flags):
    _qa_numeric_field(before, after, flags, 'number_of_reviews', 0, None, True)


def qa_review_scores_value(before, after, flags):
    _qa_numeric_field(before, after, flags, 'review_scores_value', 0, 5, False)


def qa_analysis_area(before, after, flags):
    _qa_category_field(before, after, flags, 'analysis_area', AREAS)


def qa_room_type(before, after, flags):
    _qa_category_field(before, after, flags, 'room_type', ROOM_TYPES)


def clean_price(series):
    """Bỏ '$' và ',' rồi ép sang số; giá trị không ép được thành NaN."""
    cleaned = series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def flag_score_without_reviews(after, flags):
    """Gắn cờ listing có điểm nhưng số review bằng 0."""
    has_score = after['review_scores_value_processed'].notna()
    has_no_reviews = after['number_of_reviews_processed'] == 0
    flags['score_without_reviews'] = has_score & has_no_reviews


def fillter_output(after, flags):
    out = after.copy()
    valid_num_of_nights = ~(flags["min_nights_abnormal"])
    out = out.loc[valid_num_of_nights]
    return out


def clean_listings(df: pd.DataFrame):
    """Trả về bảng trước QA, bảng sau QA và các cờ theo dòng."""
    # Bảng trước QA và bảng cờ dùng cùng chỉ mục để tra lại đúng dòng.
    before = df.reset_index(drop=True).copy()

    for field in ('analysis_area', 'room_type', 'price'):
        before[field] = clean_text(before[field])

    flags = id_flags(before, 'listing')

    after = before.copy()
    
    qa_host_id(before, flags)
    qa_number_of_reviews_ltm(before, after, flags)
    qa_number_of_reviews(before, after, flags)
    qa_review_scores_value(before, after, flags)

    qa_analysis_area(before, after, flags)
    qa_room_type(before, after, flags)

    flag_score_without_reviews(after, flags)

    out = fillter_output(after, flags)
    return before, out, flags


def clean_reviews(rv: pd.DataFrame, listings: pd.DataFrame):
    before = rv.reset_index(drop=True).copy()
    before["comments"] = before["comments"].str.strip().str.replace(r"\r?<br\s*/?>", " ", regex=True, flags=re.I)

    after = before.copy()
    after['date'] = pd.to_datetime(before["date"], format='%Y-%m-%d', errors='coerce')

    flags = id_flags(before, 'review')
    flags['review_date_missing'] = before["date"].isna()
    flags['review_date_invalid'] = before["date"].notna() & after['date'].isna()
    flags['review_outside_listings'] = ~(before['listing_id'].isin(listings["id"]))
    return before, after, flags

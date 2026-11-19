from math import inf
from pathlib import Path

import pandas as pd
import re


ROOM_TYPES = {'Entire home/apt', 'Private room'}
AREAS = {'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08'}

# ID của những chỗ ở có giá bất thường so với những gì chỗ đó có
ABNORMAL_PRICE_LISTINGS_ID = {"31186053", "688857223649708887", "49168183"}
MIN_NIGHTS_UPPER_BOUND = 365
EXCHANGE_RATE = 0.91


def kiem_tra_cau_truc_all(df_listings, df_calendar, df_reviews):
    # 1. Bảng listings
    req_listings = {"id", "host_id", "analysis_area", "price", "number_of_reviews_ltm", "number_of_reviews",
                    "review_scores_value", "room_type", "minimum_nights", "maximum_nights",
                    "calculated_host_listings_count", "estimated_occupancy_l365d", "estimated_revenue_l365d",
                    "last_review", "last_scraped"
                    }
    thieu_listings = req_listings - set(df_listings.columns)
    if thieu_listings:
        raise ValueError(f"Bảng listings thiếu cột bắt buộc: {sorted(thieu_listings)}")

    for col in ["number_of_reviews_ltm", "number_of_reviews", "review_scores_value", "minimum_nights", "maximum_nights",
                "calculated_host_listings_count", "estimated_occupancy_l365d", "estimated_revenue_l365d"]:
        if not pd.api.types.is_numeric_dtype(df_listings[col]):
            raise TypeError(f"Cột '{col}' trong bảng listings không phải là kiểu số.")

    # 2. Bảng reviews
    req_reviews = {"listing_id", "id", "date", "comments", "Mẫu LLM", "Nhãn tay"}
    thieu_reviews = req_reviews - set(df_reviews.columns)
    if thieu_reviews:
        raise ValueError(f"Bảng reviews thiếu cột bắt buộc: {sorted(thieu_reviews)}")

    # 3. Bảng calendar
    req_calendar = {"listing_id", "date", "available"}
    thieu_calendar = req_calendar - set(df_calendar.columns)
    if thieu_calendar:
        raise ValueError(f"Bảng calendar thiếu cột bắt buộc: {sorted(thieu_calendar)}")

    print("Kiểm tra cấu trúc thành công cho cả 3 bảng")


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


def qa_estimated_revenue_l365d(before, after, flags):
    _qa_numeric_field(before, after, flags, 'estimated_revenue_l365d', 0, None, False)


def qa_estimated_occupancy_l365d(before, after, flags):
    _qa_numeric_field(before, after, flags, 'estimated_occupancy_l365d', 0, 365, True)


def qa_calculated_host_listings_count(before, after, flags):
    _qa_numeric_field(before, after, flags, 'calculated_host_listings_count', 0, None, True)


def qa_analysis_area(before, after, flags):
    _qa_category_field(before, after, flags, 'analysis_area', AREAS)


def qa_room_type(before, after, flags):
    _qa_category_field(before, after, flags, 'room_type', ROOM_TYPES)


def qa_last_review(before, after, flags):
    after["last_review_processed"] = pd.to_datetime(before["last_review"], errors="coerce")

    flags["last_review_missing"] = before["last_review"].isna()
    flags["last_review_invalid"] = before["last_review"].notna() & after["last_review_processed"].isna()



def qa_last_scraped(before, after, flags):
    after["last_scraped_processed"] = pd.to_datetime(before["last_scraped"], errors="coerce")

    flags["last_scraped_missing"] = before["last_scraped"].isna()
    flags["last_scraped_invalid"] = before["last_scraped"].notna() & after["last_scraped_processed"].isna()


def clean_price(series):
    """Bỏ '$' và ',' rồi ép sang số; giá trị không ép được thành NaN."""
    cleaned = series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def price_bounds(col, price, aspect):
    """Cận IQR (dưới/trên) và P99 theo từng vùng/loại phòng (aspect), chỉ tính trên giá > 0."""
    temp_df = pd.DataFrame({aspect: col, "price": price})
    valid_mask = temp_df["price"] > 0
    valid_prices = temp_df[valid_mask]

    grp = valid_prices.groupby(aspect)["price"]
    q1_map = grp.quantile(0.25)
    q3_map = grp.quantile(0.75)
    iqr_map = q3_map - q1_map

    upper_map = q3_map + 1.5 * iqr_map
    lower_map = (q1_map - 1.5 * iqr_map).clip(lower=0)
    p99_map = grp.quantile(0.99)

    lower = temp_df[aspect].map(lower_map)
    upper = temp_df[aspect].map(upper_map)
    p99 = temp_df[aspect].map(p99_map)
    return valid_mask, lower, upper, p99


def impute_price(after):
    median_by_area_room = after.groupby(["analysis_area", "room_type"])["price_processed"].transform("median")
    after["clean_price_median_area_room"] = after["price_processed"].fillna(median_by_area_room)

    # median_by_area_room_after_excluded = (after.loc[~after["id"].isin(ABNORMAL_PRICE_LISTINGS_ID)]
    #                                     .groupby(["analysis_area", "room_type"])["price_processed"]
    #                                     .transform("median")
    #                                     )
    # after["clean_price_median_area_room"] = after["price_processed"].fillna(median_by_area_room_after_excluded)

def qa_price(before, after, flags):
    price_num = clean_price(before['price'])

    # Đổi đơn vị tiền tệ từ $ sang EUR
    price_num = round(price_num * EXCHANGE_RATE, 3)

    flags['price_missing'] = before['price'].isna()
    flags['price_parse_error'] = before['price'].notna() & price_num.isna()
    flags['price_non_positive'] = price_num.notna() & (price_num <= 0)
    flags['sus_price'] = before["id"].isin(ABNORMAL_PRICE_LISTINGS_ID)

    valid_mask, lower_a, upper_a, p99_a = price_bounds(after["analysis_area"], price_num, "analysis_area")
    flags['price_outlier_iqr_area'] = valid_mask & ((price_num > upper_a) | (price_num < lower_a))
    flags['price_outlier_p99_area'] = valid_mask & (price_num > p99_a)

    valid_mask, lower_r, upper_r, p99_r = price_bounds(after["room_type"], price_num, "room_type")
    flags['price_outlier_iqr_room_type'] = valid_mask & ((price_num > upper_r) | (price_num < lower_r))
    flags['price_outlier_p99_room_type'] = valid_mask & (price_num > p99_r)
    
    after['price_processed'] = price_num.where(valid_mask)
    impute_price(after)


def flag_score_without_reviews(after, flags):
    """Gắn cờ listing có điểm nhưng số review bằng 0."""
    has_score = after['review_scores_value_processed'].notna()
    has_no_reviews = after['number_of_reviews_processed'] == 0
    flags['score_without_reviews'] = has_score & has_no_reviews


def flag_min_max_nights(before, flags):
    flags["min_nights_abnormal"] = before["minimum_nights"].notna() & (before["minimum_nights"] > MIN_NIGHTS_UPPER_BOUND)
    flags["min_nights_greater_than_max"] = before["minimum_nights"].notna() & before["maximum_nights"].notna() & (before["minimum_nights"] > before["maximum_nights"])


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
    flag_min_max_nights(before, flags)

    after = before.copy()
    
    qa_host_id(before, flags)
    qa_number_of_reviews_ltm(before, after, flags)
    qa_number_of_reviews(before, after, flags)
    qa_review_scores_value(before, after, flags)
    qa_estimated_revenue_l365d(before, after, flags)
    qa_estimated_occupancy_l365d(before, after, flags)
    qa_calculated_host_listings_count(before, after, flags)

    qa_analysis_area(before, after, flags)
    qa_room_type(before, after, flags)

    qa_last_review(before, after, flags)
    qa_last_scraped(before, after, flags)

    qa_price(before, after, flags)
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


def clean_calendar(cal: pd.DataFrame, listings: pd.DataFrame):
    """Trả về bảng trước QA, bảng sau QA và các cờ theo dòng cho calendar."""
    before = cal.reset_index(drop=True).copy()
    after = before.copy()

    after["clean_date"] = pd.to_datetime(before["date"], errors="coerce")

    key_calendar = ["listing_id", "date"]
    valid_listing_ids = set(listings["id"].dropna())

    flags = pd.DataFrame({
        "calendar_listing_id_missing": before["listing_id"].isna(),
        "calendar_duplicate_key": before.duplicated(subset=key_calendar, keep=False),
        "calendar_date_missing": before["date"].isna(),
        "calendar_date_invalid": before["date"].notna() & after['clean_date'].isna(),
        "calendar_available_missing": before["available"].isna(),
        "calendar_available_invalid": before["available"].notna() & ~before["available"].isin(["t", "f"]),
        "calendar_outside_listings": before["listing_id"].notna() & ~before["listing_id"].isin(valid_listing_ids),
    })

    after["clean_available"] = after["available"].astype(str).str.lower().map({"t": True, "f": False})
    return before, after, flags
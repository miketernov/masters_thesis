import pandas as pd

# =========================
# 1. ПУТИ К ФАЙЛАМ
# =========================

REVIEWS_FILE = r"xxx"
RESTAURANTS_FILE = r"xxx"

OUTPUT_REVIEW_FEATURES_FILE = r"xxx"
OUTPUT_MERGED_FILE = r"xxx"

# =========================
# 2. НАЗВАНИЯ КОЛОНОК
# =========================

RESTAURANT_ID_COL = "restaurant_id"

REVIEW_LEN_CHARS_COL = "review_len_chars"
REVIEW_LEN_WORDS_COL = "review_len_words"

POSITIVE_COL = "is_positive"
NEUTRAL_COL = "is_neutral"
NEGATIVE_COL = "is_negative"

SENTIMENT_SCORE_COL = "sentiment_score"

SERVICE_COL = "service_complaint"
FOOD_COL = "food_complaint"
PRICE_COL = "price_complaint"

NEG_SERVICE_COL = "negative_service"
NEG_FOOD_COL = "negative_food"
NEG_PRICE_COL = "negative_price"

RESTAURANT_ID_COL_IN_RESTAURANTS = "restaurant_id"

# =========================
# 3. ЧТЕНИЕ CSV
# =========================

def read_csv_flexible(path):
    for enc in ["utf-8", "utf-8-sig", "cp1251"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось прочитать файл: {path}")

reviews = read_csv_flexible(REVIEWS_FILE)
restaurants = read_csv_flexible(RESTAURANTS_FILE)

print(f"reviews rows: {len(reviews)}")
print(f"restaurants rows: {len(restaurants)}")

# =========================
# 4. ПРОВЕРКА НУЖНЫХ КОЛОНОК
# =========================

required_review_cols = [
    RESTAURANT_ID_COL,
    REVIEW_LEN_CHARS_COL,
    REVIEW_LEN_WORDS_COL,
    POSITIVE_COL,
    NEUTRAL_COL,
    NEGATIVE_COL,
    SENTIMENT_SCORE_COL,
    SERVICE_COL,
    FOOD_COL,
    PRICE_COL,
    NEG_SERVICE_COL,
    NEG_FOOD_COL,
    NEG_PRICE_COL,
]

missing_review_cols = [c for c in required_review_cols if c not in reviews.columns]
if missing_review_cols:
    raise ValueError(f"В файле reviews не хватает колонок: {missing_review_cols}")

if RESTAURANT_ID_COL_IN_RESTAURANTS not in restaurants.columns:
    raise ValueError(
        f"В файле restaurants нет колонки id: {RESTAURANT_ID_COL_IN_RESTAURANTS}"
    )

# =========================
# 5. ПРИВЕДЕНИЕ ТИПОВ
# =========================

numeric_cols = [
    REVIEW_LEN_CHARS_COL,
    REVIEW_LEN_WORDS_COL,
    POSITIVE_COL,
    NEUTRAL_COL,
    NEGATIVE_COL,
    SENTIMENT_SCORE_COL,
    SERVICE_COL,
    FOOD_COL,
    PRICE_COL,
    NEG_SERVICE_COL,
    NEG_FOOD_COL,
    NEG_PRICE_COL,
]

for col in numeric_cols:
    reviews[col] = pd.to_numeric(reviews[col], errors="coerce")

reviews = reviews.dropna(subset=[RESTAURANT_ID_COL]).copy()

reviews[RESTAURANT_ID_COL] = reviews[RESTAURANT_ID_COL].astype(str)
restaurants[RESTAURANT_ID_COL_IN_RESTAURANTS] = restaurants[RESTAURANT_ID_COL_IN_RESTAURANTS].astype(str)

# =========================
# 6. АГРЕГАЦИЯ ОТЗЫВОВ ПО РЕСТОРАНУ
# =========================

review_features = (
    reviews
    .groupby(RESTAURANT_ID_COL, dropna=False)
    .agg(
        reviews_count=(RESTAURANT_ID_COL, "size"),
        avg_review_len_chars=(REVIEW_LEN_CHARS_COL, "mean"),
        avg_review_len_words=(REVIEW_LEN_WORDS_COL, "mean"),
        positive_share=(POSITIVE_COL, "mean"),
        neutral_share=(NEUTRAL_COL, "mean"),
        negative_share=(NEGATIVE_COL, "mean"),
        avg_sentiment_score=(SENTIMENT_SCORE_COL, "mean"),
        service_complaint_share=(SERVICE_COL, "mean"),
        food_complaint_share=(FOOD_COL, "mean"),
        price_complaint_share=(PRICE_COL, "mean"),
        negative_service_share=(NEG_SERVICE_COL, "mean"),
        negative_food_share=(NEG_FOOD_COL, "mean"),
        negative_price_share=(NEG_PRICE_COL, "mean"),
        negative_reviews_count=(NEGATIVE_COL, "sum"),
        negative_service_count=(NEG_SERVICE_COL, "sum"),
        negative_food_count=(NEG_FOOD_COL, "sum"),
        negative_price_count=(NEG_PRICE_COL, "sum"),
    )
    .reset_index()
)

# =========================
# 6.1 ДОПОЛНИТЕЛЬНЫЕ ФИЧИ
# =========================

def safe_divide_series(num, denom):
    result = num / denom
    result = result.where(denom > 0, 0)
    return result

review_features["service_among_negative_share"] = safe_divide_series(
    review_features["negative_service_count"],
    review_features["negative_reviews_count"]
)

review_features["food_among_negative_share"] = safe_divide_series(
    review_features["negative_food_count"],
    review_features["negative_reviews_count"]
)

review_features["price_among_negative_share"] = safe_divide_series(
    review_features["negative_price_count"],
    review_features["negative_reviews_count"]
)

# временные count-колонки можно удалить
review_features = review_features.drop(columns=[
    "negative_reviews_count",
    "negative_service_count",
    "negative_food_count",
    "negative_price_count",
])

# =========================
# 7. ОКРУГЛЕНИЕ
# =========================

# длины округляем до целого
review_features["avg_review_len_chars"] = (
    review_features["avg_review_len_chars"].round(0).astype(int)
)
review_features["avg_review_len_words"] = (
    review_features["avg_review_len_words"].round(0).astype(int)
)

# остальные показатели — до 4 знаков
round_cols = [
    "positive_share",
    "neutral_share",
    "negative_share",
    "avg_sentiment_score",
    "service_complaint_share",
    "food_complaint_share",
    "price_complaint_share",
    "negative_service_share",
    "negative_food_share",
    "negative_price_share",
    "service_among_negative_share",
    "food_among_negative_share",
    "price_among_negative_share",
]

review_features[round_cols] = review_features[round_cols].round(4)

# =========================
# 8. МЕРЖ К ОСНОВНОЙ ТАБЛИЦЕ РЕСТОРАНОВ
# =========================

restaurants_merged = restaurants.merge(
    review_features,
    how="left",
    left_on=RESTAURANT_ID_COL_IN_RESTAURANTS,
    right_on=RESTAURANT_ID_COL,
)

if (
    RESTAURANT_ID_COL != RESTAURANT_ID_COL_IN_RESTAURANTS
    and RESTAURANT_ID_COL in restaurants_merged.columns
):
    restaurants_merged = restaurants_merged.drop(columns=[RESTAURANT_ID_COL])

# =========================
# 9. ЗАПОЛНЕНИЕ ПРОПУСКОВ ДЛЯ РЕСТОРАНОВ БЕЗ ОТЗЫВОВ
# =========================

restaurants_merged["has_reviews"] = restaurants_merged["reviews_count"].notna().astype(int)

fill_zero_cols = [
    "reviews_count",
    "avg_review_len_chars",
    "avg_review_len_words",
    "positive_share",
    "neutral_share",
    "negative_share",
    "avg_sentiment_score",
    "service_complaint_share",
    "food_complaint_share",
    "price_complaint_share",
    "negative_service_share",
    "negative_food_share",
    "negative_price_share",
    "service_among_negative_share",
    "food_among_negative_share",
    "price_among_negative_share",
]

restaurants_merged[fill_zero_cols] = restaurants_merged[fill_zero_cols].fillna(0)

restaurants_merged["reviews_count"] = restaurants_merged["reviews_count"].astype(int)
restaurants_merged["avg_review_len_chars"] = restaurants_merged["avg_review_len_chars"].astype(int)
restaurants_merged["avg_review_len_words"] = restaurants_merged["avg_review_len_words"].astype(int)

# =========================
# 10. СОХРАНЕНИЕ
# =========================

review_features.to_csv(OUTPUT_REVIEW_FEATURES_FILE, index=False, encoding="utf-8-sig")
restaurants_merged.to_csv(OUTPUT_MERGED_FILE, index=False, encoding="utf-8-sig")

print("Готово.")
print(f"Файл с агрегатами отзывов: {OUTPUT_REVIEW_FEATURES_FILE}")
print(f"Файл ресторанов с review-фичами: {OUTPUT_MERGED_FILE}")

print("\nПроверка:")
print("Количество ресторанов в итоговом файле:", len(restaurants_merged))
print("Количество ресторанов с отзывами:", restaurants_merged["has_reviews"].sum())
print("Количество ресторанов без отзывов:", (restaurants_merged["has_reviews"] == 0).sum())
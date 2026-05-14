import re
import time
import random
import pandas as pd
import requests

from bs4 import BeautifulSoup
from typing import List, Optional


# =========================================================
# 1. ФАЙЛЫ
# =========================================================

INPUT_REVIEWS_FILE = r"xxx"
INPUT_RESTAURANTS_FILE = r"xxx"
OUTPUT_FILE = r"xxx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

# =========================================================
# 2. СПИСОК ПРОПУЩЕННЫХ ID И ССЫЛКИ НА ОТЗЫВЫ
# =========================================================

REVIEWS_URLS = {
    "12": "https://yandex.ru/maps/org/lyudi_v_sinem/155216285846/reviews/?indoorLevel=1&ll=37.630026%2C55.763839&z=16.6",
    "20": "https://yandex.ru/maps/org/ostanovka_gt/134475111064/reviews/?ll=37.612707%2C55.742890&z=15",
    "49": "https://yandex.ru/maps/org/chapa/207980869972/reviews/?ll=37.838145%2C55.684045&mode=search&sll=37.628971%2C55.764224&sspn=0.010284%2C0.007842&tab=reviews&text=%D0%A7%D0%B0%D0%BF%D0%B0%20%D0%B6%D1%83%D0%BA%D0%BE%D0%B2%D1%81%D0%BA%D0%BE%D0%B3%D0%BE&z=11",
    "98": "https://yandex.ru/maps/org/adventura/33939680500/reviews/?indoorLevel=1&ll=37.638844%2C55.758402&z=17",
    "118": "https://yandex.ru/maps/org/superdom/146806290019/reviews/?ll=37.632313%2C55.735459&z=17",
    "145": "https://yandex.ru/maps/org/saam/121159008218/reviews/?ll=37.617700%2C55.755874&z=10",
    "282": "https://yandex.ru/maps/org/b12/221860592357/reviews/?indoorLevel=0&ll=37.606208%2C55.756563&z=17",
    "327": "https://yandex.ru/maps/org/literaturno_alkogolny_salon_sergey_yesenin/13681112364/reviews/?indoorLevel=1&ll=37.649867%2C55.744384&z=17",
    "388": "https://yandex.ru/maps/org/tin/61883368466/reviews/?ll=37.602768%2C55.764268&z=17",
    "415": "https://yandex.ru/maps/org/yo_bistro_bar/105212869130/reviews/?indoorLevel=1&ll=37.632340%2C55.757187&z=17",
    "448": "https://yandex.ru/maps/org/lyubov_gastrobistro/177330072108/reviews/?indoorLevel=1&ll=37.634506%2C55.754927&z=17",
    "461": "https://yandex.ru/maps/org/free_co/27678670846/reviews/?ll=37.545412%2C55.746943&z=16",
    "521": "https://yandex.ru/maps/org/hungry_girl/46412081249/reviews/?ll=37.617700%2C55.755874&utm_content=add_review&utm_medium=reviews&utm_source=maps-reviews-widget&z=10",
    "548": "https://yandex.ru/maps/org/elsewhere/224827729437/reviews/?ll=37.619459%2C55.773231&z=17",
    "788": "https://yandex.ru/maps/org/rusty_rat_pizza/94015478437/reviews/?indoorLevel=1&ll=37.627786%2C55.757207&z=17",
    "876": "https://yandex.ru/maps/org/harvey_monica/68370311560/reviews/?indoorLevel=1&ll=37.646857%2C55.741531&z=17",
    "905": "https://yandex.ru/maps/org/angel_cakes_patriki/73172670175/reviews/?ll=37.595682%2C55.763786&z=16",
}

# =========================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def read_csv_flexible(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1251"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось прочитать файл: {path}")


def fetch_page(url: str, session: requests.Session) -> Optional[str]:
    try:
        response = session.get(url, headers=HEADERS, timeout=25)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"[ERROR] fetch_page {url}: {e}")
        return None


def clean_review_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_reviews_from_dom(html: str, limit: int = 50) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    reviews = []

    selectors = [
        "span.spoiler-view__text-container",
        "span.business-review-view__body-text",
        "div.business-review-view__body",
        "div.business-review-view__expandable",
    ]

    for selector in selectors:
        for tag in soup.select(selector):
            text = clean_review_text(tag.get_text(" ", strip=True))
            if text and text not in reviews and len(text) >= 10:
                reviews.append(text)
            if len(reviews) >= limit:
                return reviews

    return reviews


def extract_reviews_from_scripts(html: str, limit: int = 50) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    reviews = []

    patterns = [
        r'"text"\s*:\s*"((?:[^"\\]|\\.){15,}?)"',
        r'"reviewText"\s*:\s*"((?:[^"\\]|\\.){15,}?)"',
        r'"body"\s*:\s*"((?:[^"\\]|\\.){15,}?)"',
    ]

    for script in soup.find_all("script"):
        content = script.string or script.get_text()
        if not content:
            continue

        for pattern in patterns:
            matches = re.findall(pattern, content)

            for match in matches:
                try:
                    text = bytes(match, "utf-8").decode("unicode_escape")
                except Exception:
                    text = match

                text = (
                    text.replace('\\"', '"')
                    .replace("\\n", " ")
                    .replace("\\t", " ")
                    .replace("\\r", " ")
                )
                text = clean_review_text(text)

                if text and text not in reviews and len(text) >= 10:
                    reviews.append(text)

                if len(reviews) >= limit:
                    return reviews

    return reviews


def get_reviews(reviews_url: str, session: requests.Session, limit: int = 50) -> List[str]:
    html = fetch_page(reviews_url, session)
    if not html:
        return []

    reviews = extract_reviews_from_dom(html, limit=limit)
    if not reviews:
        reviews = extract_reviews_from_scripts(html, limit=limit)

    return reviews[:limit]


def add_missing_columns(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    return df


# =========================================================
# 4. ОСНОВНАЯ ЛОГИКА
# =========================================================

def main():
    existing_reviews = read_csv_flexible(INPUT_REVIEWS_FILE)
    restaurants = read_csv_flexible(INPUT_RESTAURANTS_FILE)

    # унифицируем ID
    if "restaurant_id" not in existing_reviews.columns:
        raise ValueError("В existing review file нет колонки restaurant_id")

    if "restaurant_id" not in restaurants.columns:
        if "id" in restaurants.columns:
            restaurants = restaurants.rename(columns={"id": "restaurant_id"})
        else:
            raise ValueError("В restaurants file нет колонки restaurant_id или id")

    existing_reviews["restaurant_id"] = existing_reviews["restaurant_id"].astype(str)
    restaurants["restaurant_id"] = restaurants["restaurant_id"].astype(str)

    # Проверим, какие из нужных ID уже случайно есть
    existing_ids = set(existing_reviews["restaurant_id"].dropna().astype(str).unique())
    target_ids = set(REVIEWS_URLS.keys())
    really_missing_ids = sorted(list(target_ids - existing_ids), key=lambda x: int(x))

    print("ID, которые реально будут допарсены:")
    print(really_missing_ids)

    if not really_missing_ids:
        print("Все целевые restaurant_id уже есть в existing review file.")
        result = existing_reviews.copy()

        drop_cols = [c for c in ["search_url", "org_url", "reviews_url"] if c in result.columns]
        if drop_cols:
            result = result.drop(columns=drop_cols)

        result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"Готово: {OUTPUT_FILE}")
        return

    # подтянем инфу о ресторанах только для пропусков
    missing_restaurants = restaurants[restaurants["restaurant_id"].isin(really_missing_ids)].copy()

    if missing_restaurants.empty:
        raise ValueError("В restaurant file не найдены строки по пропущенным restaurant_id")

    session = requests.Session()
    new_rows = []

    # Чтобы подстроиться под формат исходного review-файла
    required_cols = list(existing_reviews.columns)
    base_cols = set(required_cols)

    for _, row in missing_restaurants.iterrows():
        restaurant_id = str(row["restaurant_id"]).strip()
        rest_name = str(row.get("rest_name", "")).strip()
        full_address = str(row.get("full_address", "")).strip()
        reviews_url = REVIEWS_URLS.get(restaurant_id)

        if not reviews_url:
            print(f"[SKIP] id={restaurant_id}: ссылка не задана")
            continue

        reviews = get_reviews(reviews_url, session=session, limit=50)

        print(
            f"id={restaurant_id} | {rest_name} | "
            f"reviews_found={len(reviews)}"
        )

        for review_num, review_text in enumerate(reviews, start=1):
            row_dict = {col: None for col in required_cols}

            # Заполняем базовые поля, если они существуют в старом файле
            if "restaurant_id" in base_cols:
                row_dict["restaurant_id"] = restaurant_id
            if "rest_name" in base_cols:
                row_dict["rest_name"] = rest_name
            if "full_address" in base_cols:
                row_dict["full_address"] = full_address
            if "review_num" in base_cols:
                row_dict["review_num"] = review_num
            if "review_text" in base_cols:
                row_dict["review_text"] = review_text

            # Если вдруг в старом файле еще были такие колонки — можно сразу заполнить
            if "reviews_url" in base_cols:
                row_dict["reviews_url"] = reviews_url

            new_rows.append(row_dict)

        time.sleep(random.uniform(2, 4))

    new_reviews = pd.DataFrame(new_rows)

    # Гарантируем одинаковые колонки
    new_reviews = add_missing_columns(new_reviews, list(existing_reviews.columns))
    new_reviews = new_reviews[existing_reviews.columns]

    # Склеиваем
    result = pd.concat([existing_reviews, new_reviews], ignore_index=True)

    # Удаляем технические колонки, как ты просил
    drop_cols = [c for c in ["search_url", "org_url", "reviews_url"] if c in result.columns]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    # Убираем полные дубли, если вдруг что-то совпало
    subset_cols = [c for c in ["restaurant_id", "review_text"] if c in result.columns]
    if subset_cols:
        before = len(result)
        result = result.drop_duplicates(subset=subset_cols)
        after = len(result)
        print(f"Удалено дублей: {before - after}")

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nГотово.")
    print(f"Итоговый файл: {OUTPUT_FILE}")
    print(f"Было строк: {len(existing_reviews)}")
    print(f"Добавлено новых строк: {len(new_reviews)}")
    print(f"Стало строк после dedup: {len(result)}")


if __name__ == "__main__":
    main()
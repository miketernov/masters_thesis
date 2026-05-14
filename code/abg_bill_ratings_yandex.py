from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import time
from urllib.parse import quote

INPUT_FILE = r"xxx"
OUTPUT_FILE = r"xxx"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# -------------------- AVERAGE BILL --------------------
def extract_average_bill(soup: BeautifulSoup):
    elements = soup.select("span.business-features-view__valued-value")

    for el in elements:
        text = el.get_text(" ", strip=True).replace("\xa0", " ")

        if "₽" not in text:
            continue

        # от 2000 ₽
        m = re.search(r"от\s*(\d+(?:\s?\d+)*)\s*₽", text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1).replace(" ", ""))

        # 1500–2500 ₽
        m = re.search(r"(\d+(?:\s?\d+)*)\s*[–—-]\s*(\d+(?:\s?\d+)*)\s*₽", text)
        if m:
            low = int(m.group(1).replace(" ", ""))
            high = int(m.group(2).replace(" ", ""))
            return int((low + high) / 2)

        # 1500 ₽
        m = re.search(r"(\d+(?:\s?\d+)*)\s*₽", text)
        if m:
            return int(m.group(1).replace(" ", ""))

    return None


# -------------------- RATING --------------------
def extract_rating(soup: BeautifulSoup):
    el = soup.select_one("span.business-rating-badge-view__rating-text")
    if el:
        return float(el.get_text(strip=True).replace(",", "."))
    return None


# -------------------- RATINGS COUNT --------------------
def extract_ratings_count(soup: BeautifulSoup):
    el = soup.select_one("div.business-header-rating-view__text._clickable")
    if not el:
        return None

    text = el.get_text(" ", strip=True).replace("\xa0", " ")

    match = re.search(r"(\d[\d\s]*)", text)
    if match:
        return int(match.group(1).replace(" ", ""))

    return None


# -------------------- REVIEWS COUNT --------------------
def extract_reviews_count(soup: BeautifulSoup):
    elements = soup.select(".tabs-select-view__title._name_reviews")

    for el in elements:
        text = el.get_text(" ", strip=True).replace("\xa0", " ")

        match = re.search(r"(\d[\d\s]*)", text)
        if match:
            return int(match.group(1).replace(" ", ""))

    return None


# -------------------- URL --------------------
def build_url(name, coords):
    return f"https://yandex.ru/maps/?text={quote(str(name))}&ll={coords}&z=17"


# -------------------- PARSER --------------------
def parse_one(name, coords):
    url = build_url(name, coords)

    try:
        html = requests.get(url, headers=HEADERS, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        return {
            "url": url,
            "rating": None,
            "ratings_count": None,
            "reviews_count": None,
            "average_bill": None,
            "error": str(e)
        }

    return {
        "url": url,
        "rating": extract_rating(soup),
        "ratings_count": extract_ratings_count(soup),
        "reviews_count": extract_reviews_count(soup),
        "average_bill": extract_average_bill(soup),
        "error": None
    }


# -------------------- MAIN --------------------
def main():
    df = pd.read_csv(INPUT_FILE)

    results = []

    for i, row in df.iterrows():
        name = row["rest_name"]
        coords = row["coordinates"]

        print(f"[{i+1}/{len(df)}] {name}")

        data = parse_one(name, coords)
        results.append(data)

        time.sleep(1.2)  # важно чтобы не забанили

    result_df = pd.DataFrame(results)

    final_df = pd.concat([df, result_df], axis=1)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("ГОТОВО:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
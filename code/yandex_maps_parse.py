from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
from urllib.parse import quote

INPUT_FILE = r"xxx"
TEST_INDEX = 0  # какую строку из CSV проверить

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def extract_average_bill(soup: BeautifulSoup):
    elements = soup.select("span.business-features-view__valued-value")

    print("\n--- Блок business-features-view__valued-value ---")
    print("Найдено:", len(elements))

    for i, el in enumerate(elements):
        text = el.get_text(" ", strip=True).replace("\xa0", " ")
        print(f"{i}: '{text}'")

        if "₽" not in text:
            continue

        m = re.search(r"от\s*(\d+(?:\s?\d+)*)\s*₽", text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1).replace(" ", ""))

        m = re.search(r"(\d+(?:\s?\d+)*)\s*[–—-]\s*(\d+(?:\s?\d+)*)\s*₽", text)
        if m:
            low = int(m.group(1).replace(" ", ""))
            high = int(m.group(2).replace(" ", ""))
            return int((low + high) / 2)

        m = re.search(r"(\d+(?:\s?\d+)*)\s*₽", text)
        if m:
            return int(m.group(1).replace(" ", ""))

    return None


def extract_rating(soup: BeautifulSoup):
    el = soup.select_one("span.business-rating-badge-view__rating-text")

    print("\n--- rating ---")
    if el:
        return float(el.get_text(strip=True).replace(",", "."))
    return None


def extract_ratings_count(soup: BeautifulSoup):
    el = soup.select_one("div.business-header-rating-view__text _clickable")

    print("\n--- ratings_count ---")
    if not el:
        print("Не найден")
        return None

    text = el.get_text(" ", strip=True).replace("\xa0", " ")
    print("RAW:", repr(text))

    match = re.search(r"(\d[\d\s]*)", text)
    if match:
        return int(match.group(1).replace(" ", ""))

    return None


def extract_reviews_count(soup: BeautifulSoup):
    elements = soup.select(".tabs-select-view__title._name_reviews")

    print("\n--- reviews_count ---")
    print("Найдено элементов:", len(elements))

    for i, el in enumerate(elements):
        text = el.get_text(" ", strip=True).replace("\xa0", " ")
        print(f"{i}: RAW: {repr(text)}")

        match = re.search(r"(\d[\d\s]*)", text)
        if match:
            return int(match.group(1).replace(" ", ""))

    return None


def build_url(name, coords):
    return f"https://yandex.ru/maps/?text={quote(str(name))}&ll={coords}&z=17"


def main():
    df = pd.read_csv(INPUT_FILE)

    row = df.iloc[TEST_INDEX]
    name = row["rest_name"]
    coords = row["coordinates"]

    print("=== ТЕСТОВАЯ СТРОКА ===")
    print("rest_name:", name)
    print("coordinates:", coords)

    url = build_url(name, coords)
    print("url:", url)

    html = requests.get(url, headers=HEADERS, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")

    rating = extract_rating(soup)
    ratings_count = extract_ratings_count(soup)
    reviews_count = extract_reviews_count(soup)
    average_bill = extract_average_bill(soup)

    print("\n=== ИТОГ ===")
    print("rating =", rating)
    print("ratings_count =", ratings_count)
    print("reviews_count =", reviews_count)
    print("average_bill =", average_bill)


if __name__ == "__main__":
    main()
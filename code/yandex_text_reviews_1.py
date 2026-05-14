import re
import time
import random
import pandas as pd
import requests

from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin
from typing import Optional, List, Tuple


INPUT_FILE = r"xxx"
OUTPUT_FILE = "xxx"

BASE_SEARCH_URL = "https://yandex.ru/maps/213/moscow/search/"
BASE_DOMAIN = "https://yandex.ru"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


def fetch_page(url: str, session: requests.Session) -> Optional[str]:
    try:
        response = session.get(url, headers=HEADERS, timeout=25)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"[ERROR] fetch_page {url}: {e}")
        return None


def build_search_url(rest_name: str, full_address: str) -> str:
    query = f"{rest_name} {full_address}".strip()
    return f"{BASE_SEARCH_URL}{quote_plus(query)}"


def extract_org_url(search_html: str) -> Optional[str]:
    """
    Пытаемся найти ссылку на карточку организации вида:
    /maps/org/<slug>/<id>/
    """
    soup = BeautifulSoup(search_html, "html.parser")

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]

        if "/org/" not in href:
            continue

        full_url = urljoin(BASE_DOMAIN, href)

        full_url = re.sub(r"/reviews/?(\?.*)?$", "", full_url)
        full_url = full_url.rstrip("/") + "/"

        if re.search(r"/org/[^/]+/\d+/", full_url):
            return full_url

    patterns = [
        r'(https://yandex\.ru/maps/org/[^"\']+/\d+/?)',
        r'(https://yandex\.ru/org/[^"\']+/\d+/?)',
        r'(/maps/org/[^"\']+/\d+/?)',
        r'(/org/[^"\']+/\d+/?)',
    ]

    for pattern in patterns:
        match = re.search(pattern, search_html)
        if match:
            full_url = urljoin(BASE_DOMAIN, match.group(1))
            full_url = re.sub(r"/reviews/?(\?.*)?$", "", full_url)
            full_url = full_url.rstrip("/") + "/"
            return full_url

    return None


def make_reviews_url(org_url: str) -> str:
    return org_url.rstrip("/") + "/reviews/"


def clean_review_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_reviews_from_dom(html: str, limit: int = 50) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    reviews = []

    for tag in soup.select("span.spoiler-view__text-container"):
        text = clean_review_text(tag.get_text(" ", strip=True))
        if text and text not in reviews:
            reviews.append(text)
        if len(reviews) >= limit:
            break

    return reviews


def extract_reviews_from_scripts(html: str, limit: int = 50) -> List[str]:
    """
    Фолбэк: иногда тексты лежат в script.
    """
    soup = BeautifulSoup(html, "html.parser")
    reviews = []

    for script in soup.find_all("script"):
        content = script.string or script.get_text()
        if not content:
            continue

        matches = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.){15,}?)"', content)

        for match in matches:
            try:
                text = bytes(match, "utf-8").decode("unicode_escape")
            except Exception:
                text = match

            text = (
                text.replace('\\"', '"')
                .replace("\\n", " ")
                .replace("\\t", " ")
            )
            text = clean_review_text(text)

            if text and text not in reviews:
                reviews.append(text)

            if len(reviews) >= limit:
                return reviews

    return reviews


def get_reviews_by_query(
    rest_name: str,
    full_address: str,
    session: requests.Session,
    limit: int = 50
) -> Tuple[str, Optional[str], Optional[str], List[str]]:
    search_url = build_search_url(rest_name, full_address)
    search_html = fetch_page(search_url, session=session)

    if not search_html:
        return search_url, None, None, []

    org_url = extract_org_url(search_html)
    if not org_url:
        return search_url, None, None, []

    reviews_url = make_reviews_url(org_url)
    reviews_html = fetch_page(reviews_url, session=session)

    if not reviews_html:
        return search_url, org_url, reviews_url, []

    reviews = extract_reviews_from_dom(reviews_html, limit=limit)

    if not reviews:
        reviews = extract_reviews_from_scripts(reviews_html, limit=limit)

    return search_url, org_url, reviews_url, reviews[:limit]


def main():
    df = pd.read_csv(INPUT_FILE)

    required_cols = ["rest_name", "full_address"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"В файле нет обязательной колонки: {col}")

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    session = requests.Session()
    output_rows = []

    for i, row in df.iterrows():
        restaurant_id = row["id"]
        rest_name = str(row.get("rest_name", "")).strip()
        full_address = str(row.get("full_address", "")).strip()

        if (
            not rest_name or
            not full_address or
            rest_name.lower() == "nan" or
            full_address.lower() == "nan"
        ):
            print(f"[SKIP] row={i}, id={restaurant_id}: пустые rest_name/full_address")
            continue

        search_url, org_url, reviews_url, reviews = get_reviews_by_query(
            rest_name=rest_name,
            full_address=full_address,
            session=session,
            limit=50
        )

        print(
            f"[{i}] id={restaurant_id} | {rest_name} | "
            f"org_found={org_url is not None} | reviews_found={len(reviews)}"
        )

        for review_num, review_text in enumerate(reviews, start=1):
            output_rows.append({
                "restaurant_id": restaurant_id,
                "rest_name": rest_name,
                "full_address": full_address,
                "search_url": search_url,
                "org_url": org_url,
                "reviews_url": reviews_url,
                "review_num": review_num,
                "review_text": review_text,
            })

        time.sleep(random.uniform(2, 4))

    reviews_df = pd.DataFrame(output_rows)

    if reviews_df.empty:
        print("Отзывы не были найдены ни для одного ресторана.")
        reviews_df = pd.DataFrame(columns=[
            "restaurant_id",
            "rest_name",
            "full_address",
            "search_url",
            "org_url",
            "reviews_url",
            "review_num",
            "review_text",
        ])

    reviews_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nГотово: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
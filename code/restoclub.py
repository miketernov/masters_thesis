import csv
import re
import time
from datetime import date
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.restoclub.ru"
TARGET_LINKS_COUNT = 500
START_ID = 554
OUTPUT_CSV = "restaurants_restoclub.csv"

SEARCH_URL_FIRST = (
    "https://www.restoclub.ru/msk/search"
    "?sort=cost&direction=desc&expertChoice=false"
    "&types%5B%5D=3&types%5B%5D=30&types%5B%5D=10&types%5B%5D=23&types%5B%5D=2"
    "&types%5B%5D=20&types%5B%5D=14&types%5B%5D=11&types%5B%5D=4&types%5B%5D=47"
    "&types%5B%5D=15&types%5B%5D=1&types%5B%5D=17&types%5B%5D=37&types%5B%5D=13"
    "&types%5B%5D=18&polygons%5B%5D=341&polygons%5B%5D=154&polygons%5B%5D=312"
    "&polygons%5B%5D=340&polygons%5B%5D=377&polygons%5B%5D=342&polygons%5B%5D=322"
    "&polygons%5B%5D=7&polygons%5B%5D=357&polygons%5B%5D=148&polygons%5B%5D=305"
    "&polygons%5B%5D=112&polygons%5B%5D=117"
)

SEARCH_URL_PAGED = (
    "https://www.restoclub.ru/msk/search/{page}"
    "?sort=cost&direction=desc&expertChoice=false"
    "&types%5B%5D=3&types%5B%5D=30&types%5B%5D=10&types%5B%5D=23&types%5B%5D=2"
    "&types%5B%5D=20&types%5B%5D=14&types%5B%5D=11&types%5B%5D=4&types%5B%5D=47"
    "&types%5B%5D=15&types%5B%5D=1&types%5B%5D=17&types%5B%5D=37&types%5B%5D=13"
    "&types%5B%5D=18&polygons%5B%5D=341&polygons%5B%5D=154&polygons%5B%5D=312"
    "&polygons%5B%5D=340&polygons%5B%5D=377&polygons%5B%5D=342&polygons%5B%5D=322"
    "&polygons%5B%5D=7&polygons%5B%5D=357&polygons%5B%5D=148&polygons%5B%5D=305"
    "&polygons%5B%5D=112&polygons%5B%5D=117"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
}

PLACE_PATH_RE = re.compile(r"^/msk/place/[^/?#]+/?$")


def get_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_html(session, url, timeout=30):
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def clean_text(text):
    if not text:
        return ""
    return " ".join(text.strip().split())


def build_search_url(page):
    if page == 1:
        return SEARCH_URL_FIRST
    return SEARCH_URL_PAGED.format(page=page)


def normalize_place_url(href):
    if not href:
        return None

    full_url = urljoin(BASE_URL, href)
    parsed = urlparse(full_url)
    path = parsed.path.rstrip("/")

    if not PLACE_PATH_RE.match(path):
        return None

    return "{}://{}{}".format(parsed.scheme, parsed.netloc, path)


def parse_place_links(html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.select("a.search-place-title__link.search-place-title__link-name"):
        href = a.get("data-href") or a.get("href")
        normalized = normalize_place_url(href)
        if normalized:
            links.add(normalized)

    return sorted(links)


def extract_first_text(soup, selectors):
    for selector in selectors:
        el = soup.select_one(selector)
        if el:
            txt = clean_text(el.get_text(" ", strip=True))
            if txt:
                return txt
    return ""


def parse_place_page(html, url):
    soup = BeautifulSoup(html, "html.parser")

    name = extract_first_text(soup, [
        "h1.PlaceInfo_Title__o_AA7",
        "h1[class*='PlaceInfo_Title']",
        "h1",
    ])

    metro = extract_first_text(soup, [
        "span.Subway_SubwayName__QK0VK",
        "span[class*='Subway_SubwayName']",
    ])

    address = extract_first_text(soup, [
        "span.Address_AddressTitleLink__p_xqk",
        "span[class*='Address_AddressTitleLink']",
        "a[class*='Address_AddressTitleLink']",
    ])

    return {
        "url": url,
        "name": name,
        "metro": metro,
        "address": address,
    }


def collect_place_links(session, target_count=500, max_pages=100):
    collected = []
    seen = set()

    for page in range(1, max_pages + 1):
        search_url = build_search_url(page)
        print("[SEARCH] page={} -> {}".format(page, search_url))

        try:
            html = fetch_html(session, search_url)
        except Exception as e:
            print("[ERROR] Не удалось открыть страницу поиска {}: {}".format(page, e))
            break

        page_links = parse_place_links(html)
        print("[INFO] Найдено канонических ссылок на странице: {}".format(len(page_links)))

        if not page_links:
            print("[INFO] На странице нет новых карточек, останавливаюсь.")
            break

        new_count = 0
        for link in page_links:
            if link not in seen:
                seen.add(link)
                collected.append(link)
                new_count += 1

                if len(collected) >= target_count:
                    print("[INFO] Собрано {} ссылок.".format(len(collected)))
                    return collected

        print("[INFO] Новых ссылок добавлено: {}, всего: {}".format(new_count, len(collected)))
        time.sleep(1.0)

    return collected


def collect_places_data(session, links):
    rows = []
    today = str(date.today())

    for idx, link in enumerate(links, start=START_ID):
        print("[PLACE] {} -> {}".format(idx, link))

        try:
            html = fetch_html(session, link)
            place = parse_place_page(html, link)
        except Exception as e:
            print("[ERROR] Не удалось обработать {}: {}".format(link, e))
            place = {
                "url": link,
                "name": "",
                "metro": "",
                "address": "",
            }

        rows.append({
            "id": idx,
            "date": today,
            "status": "active",
            "district": place["metro"],
            "text": "restoclub",
            "название_заведения": place["name"],
            "адрес": place["address"],
            "срок_работы_в_годах": 2,
        })

        time.sleep(0.8)

    return rows


def save_to_csv(rows, filename):
    fieldnames = [
        "id",
        "date",
        "status",
        "district",
        "text",
        "название_заведения",
        "адрес",
        "срок_работы_в_годах",
    ]

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    session = get_session()

    links = collect_place_links(session, target_count=TARGET_LINKS_COUNT, max_pages=100)
    print("[DONE] Всего собрано ссылок: {}".format(len(links)))

    if not links:
        print("[STOP] Не удалось собрать ссылки.")
        return

    rows = collect_places_data(session, links)
    save_to_csv(rows, OUTPUT_CSV)

    print("[DONE] CSV сохранен: {}".format(OUTPUT_CSV))


if __name__ == "__main__":
    main()
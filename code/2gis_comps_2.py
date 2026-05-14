import math
import requests
import pandas as pd

API_KEY = "xxx"
INPUT_FILE = r"xxx"
OUTPUT_FILE = r"xxx"

SEARCH_URL = "https://catalog.api.2gis.com/3.0/items"

START_ID = 1


COMPETITOR_GROUPS = {
    "кафе": ["кафе", "ресторан", "пиццерия", "кофейня", "быстроепитание"],
    "ресторан": ["ресторан", "кафе", "пиццерия", "бар"],
    "пиццерия": ["пиццерия", "кафе", "ресторан", "быстроепитание"],
    "бар": ["бар", "паб", "винотека", "кальянная", "клуб", "караокеклуб"],
    "паб": ["паб", "бар", "винотека", "клуб", "караокеклуб"],
    "клуб": ["клуб", "караокеклуб", "бар", "паб", "кальянная"],
    "караокеклуб": ["караокеклуб", "клуб", "бар", "паб", "кальянная"],
    "быстроепитание": ["быстроепитание", "кафе", "пиццерия", "кофейня"],
    "кофейня": ["кофейня", "пекарня", "кафе", "быстроепитание"],
    "пекарня": ["пекарня", "кофейня", "кафе"],
    "винотека": ["винотека", "бар", "паб", "ресторан"],
    "кальянная": ["кальянная", "бар", "клуб", "караокеклуб", "паб"],
}

SEARCH_QUERY_MAP = {
    "кафе": "кафе",
    "ресторан": "ресторан",
    "пиццерия": "пиццерия",
    "бар": "бар",
    "паб": "паб",
    "клуб": "клуб",
    "караокеклуб": "караоке",
    "быстроепитание": "быстрое питание",
    "кофейня": "кофейня",
    "пекарня": "пекарня",
    "винотека": "винотека",
    "кальянная": "кальянная",
}


def haversine(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_coordinates(coord_value):
    if pd.isna(coord_value):
        return None, None

    parts = str(coord_value).split(",")
    if len(parts) != 2:
        return None, None

    try:
        lon = float(parts[0].strip())
        lat = float(parts[1].strip())
        return lat, lon
    except ValueError:
        return None, None


def normalize_rest_type(rest_type):
    if pd.isna(rest_type):
        return None
    return str(rest_type).strip().lower().replace(" ", "")


def get_search_queries_for_rest_type(rest_type):
    norm_type = normalize_rest_type(rest_type)

    if norm_type not in COMPETITOR_GROUPS:
        return []

    competitor_types = COMPETITOR_GROUPS[norm_type]

    queries = []
    for comp_type in competitor_types:
        q = SEARCH_QUERY_MAP.get(comp_type)
        if q:
            queries.append(q)

    return list(dict.fromkeys(queries))


def fetch_competitor_items_300m(session, lat, lon, search_queries):
    all_items = {}

    for food_q in search_queries:
        for page in range(1, 100):
            params = {
                "q": food_q,
                "type": "branch",
                "point": f"{lon},{lat}",
                "radius": 300,
                "fields": "items.id,items.reviews,items.name,items.point,items.address_name",
                "page_size": 10,
                "page": page,
                "key": API_KEY,
            }

            r = session.get(SEARCH_URL, params=params, timeout=30)
            r.raise_for_status()

            items = r.json().get("result", {}).get("items", [])
            if not items:
                break

            for item in items:
                item_id = item.get("id")
                point = item.get("point", {})
                item_lat = point.get("lat")
                item_lon = point.get("lon")

                if item_id and item_lat is not None and item_lon is not None:
                    all_items[item_id] = item

            if len(items) < 10:
                break

    return list(all_items.values())


def calc_stats_by_radius(center_lat, center_lon, items, radius):
    count = 0
    ratings = []

    for item in items:
        point = item.get("point", {})
        item_lat = point.get("lat")
        item_lon = point.get("lon")

        if item_lat is None or item_lon is None:
            continue

        dist = haversine(center_lat, center_lon, item_lat, item_lon)

        if dist <= radius:
            count += 1

            rating = item.get("reviews", {}).get("general_rating")
            if rating is not None:
                try:
                    ratings.append(float(rating))
                except (TypeError, ValueError):
                    pass

    avg_rating = sum(ratings) / len(ratings) if ratings else None
    return count, avg_rating


df = pd.read_csv(INPUT_FILE)

if "id" not in df.columns:
    raise ValueError("В файле нет колонки 'id'")

df["id"] = pd.to_numeric(df["id"], errors="coerce")

target_cols = [
    "comp_100m",
    "comp_200m",
    "comp_300m",
    "comp_rating_avg_100m",
    "comp_rating_avg_200m",
    "comp_rating_avg_300m",
]

for col in target_cols:
    if col not in df.columns:
        df[col] = None

session = requests.Session()
cache = {}

# Только строки после START_ID
mask_id = df["id"] >= START_ID

# Только строки, где все 6 целевых полей пустые
mask_empty = df[target_cols].isna().all(axis=1)

# Если хочешь учитывать и пустые строки "", можно использовать так:
# mask_empty = df[target_cols].replace("", pd.NA).isna().all(axis=1)

rows_to_process = df[mask_id & mask_empty]

print(f"Всего строк в файле: {len(df)}")
print(f"Будет обработано строк: {len(rows_to_process)}")
print(f"Условие: id >= {START_ID} и все поля конкурентов пустые")

for n, (i, row) in enumerate(rows_to_process.iterrows(), start=1):
    row_id = row.get("id")
    coord_value = row.get("coordinates")
    rest_type = row.get("rest_type")

    print(f"[{n}/{len(rows_to_process)}] id={row_id} | coordinates={coord_value} | rest_type={rest_type}")

    try:
        lat, lon = parse_coordinates(coord_value)

        if lat is None or lon is None:
            print("  -> некорректные coordinates")
            continue

        search_queries = get_search_queries_for_rest_type(rest_type)

        if not search_queries:
            print("  -> неизвестный rest_type, пропуск")
            continue

        cache_key = f"{round(lat, 5)}_{round(lon, 5)}|{'|'.join(search_queries)}"

        if cache_key in cache:
            items_300 = cache[cache_key]
        else:
            items_300 = fetch_competitor_items_300m(session, lat, lon, search_queries)
            cache[cache_key] = items_300

        c100, r100 = calc_stats_by_radius(lat, lon, items_300, 100)
        c200, r200 = calc_stats_by_radius(lat, lon, items_300, 200)
        c300, r300 = calc_stats_by_radius(lat, lon, items_300, 300)

        df.at[i, "comp_100m"] = c100
        df.at[i, "comp_200m"] = c200
        df.at[i, "comp_300m"] = c300

        df.at[i, "comp_rating_avg_100m"] = round(r100, 4) if r100 is not None else None
        df.at[i, "comp_rating_avg_200m"] = round(r200, 4) if r200 is not None else None
        df.at[i, "comp_rating_avg_300m"] = round(r300, 4) if r300 is not None else None

        print(
            f"  -> queries={search_queries} | "
            f"100м: {c100}, avg={r100} | "
            f"200м: {c200}, avg={r200} | "
            f"300м: {c300}, avg={r300}"
        )

    except Exception as e:
        print(f"  -> ошибка: {e}")

    if n % 20 == 0:
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"Промежуточно сохранено после {n} обработанных строк")

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Готово. Файл сохранен: {OUTPUT_FILE}")
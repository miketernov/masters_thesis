import pandas as pd
import requests

INPUT_FILE = "C:\\Users\\test_\\PycharmProjects\\DIPLOMA_MAGA_TEST\\restaurants_final_v3.csv"
OUTPUT_FILE = "restaurants_with_coordinates.csv"

GEOCODE_URL = "https://catalog.api.2gis.com/3.0/items/geocode"
MOSCOW_CITY_ID = "4504222397630173"
API_KEY = "xxx"


def geocode_address(address: str):
    query = f"Москва, {address}"

    params = {
        "q": query,
        "city_id": MOSCOW_CITY_ID,
        "fields": "items.point",
        "key": API_KEY,
    }

    r = requests.get(GEOCODE_URL, params=params, timeout=20)
    r.raise_for_status()

    items = r.json().get("result", {}).get("items", [])
    if not items:
        return None

    point = items[0].get("point", {})
    lat = point.get("lat")
    lon = point.get("lon")

    if lat is None or lon is None:
        return None

    return lat, lon


# ======================
# MAIN
# ======================

df = pd.read_csv(INPUT_FILE)

# добавляем колонку если нет
if "coordinates" not in df.columns:
    df["coordinates"] = None

for i, row in df.iterrows():
    address = row["full_address"]

    print(f"[{i + 1}/{len(df)}] {address}")

    try:
        if pd.isna(address) or str(address).strip() == "":
            continue

        coords = geocode_address(str(address).strip())

        if coords is None:
            print("  -> не найдено")
            continue

        lat, lon = coords
        df.at[i, "coordinates"] = f"{lon},{lat}"

        print(f"  -> {lon},{lat}")

    except Exception as e:
        print(f"  -> ошибка: {e}")

    # сохраняем каждые 20 строк (чтобы не потерять прогресс)
    if (i + 1) % 20 == 0:
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"Промежуточно сохранено: {i + 1}")

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print("Готово")
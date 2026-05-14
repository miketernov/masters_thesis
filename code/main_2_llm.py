import pandas as pd
import re

INPUT_FILE = "telegram_2026_03_01_13_filtered.csv"
OUTPUT_FILE = "telegram_2026_03_parsed.csv"

df = pd.read_csv(INPUT_FILE)

# ----------------------------
# Regex-паттерны для адреса
# ----------------------------
ADDRESS_PATTERNS = [
    # "по адресу Складочная улица, 1с33"
    re.compile(
        r'по адресу\s+([А-ЯЁA-Za-zа-яё0-9\s\-/.,]+?\d+[А-ЯЁA-Za-zа-яё0-9/\-]*)',
        re.IGNORECASE
    ),

    # "(Сходненская улица, 56)"
    re.compile(
        r'\(([А-ЯЁA-Za-zа-яё0-9\s\-/.,]+?\d+[А-ЯЁA-Za-zа-яё0-9/\-]*)\)',
        re.IGNORECASE
    ),

    # "на Митинской улице, 41 открылся ..."
    re.compile(
        r'на\s+([А-ЯЁA-Za-zа-яё0-9\s\-]+?(?:улице|улица|улицы|проспекте|проспект|переулке|переулок|'
        r'бульваре|бульвар|набережной|шоссе|проезде|проезд|площади),?\s*\d+[А-ЯЁA-Za-zа-яё0-9/\-]*)',
        re.IGNORECASE
    ),

    # "в Пожарском переулке, 3 ..."
    re.compile(
        r'в\s+([А-ЯЁA-Za-zа-яё0-9\s\-]+?(?:улице|улица|улицы|проспекте|проспект|переулке|переулок|'
        r'бульваре|бульвар|набережной|шоссе|проезде|проезд|площади),?\s*\d+[А-ЯЁA-Za-zа-яё0-9/\-]*)',
        re.IGNORECASE
    ),
]

# ----------------------------
# Regex-паттерны для названия
# ----------------------------
NAME_PATTERNS = [
    # "ресторан «Банега»", "кофейня «Блин, кофе!»"
    re.compile(
        r'(?:ресторан|кофейня|кафе-кондитерская|кафе|корнер|бар|пиццерия|бистро|чайхона|'
        r'кондитерская|пекарня|бургерная|шаурма)\s+[«"]([^«»"]+)[»"]',
        re.IGNORECASE
    ),

    # "«Шоколадница» ... закрылась"
    re.compile(
        r'[«"]([^«»"]+)[»"]',
        re.IGNORECASE
    ),

    # "... открылся PIMS."
    re.compile(
        r'(?:открылся|открылась|открылось|закрылся|закрылась|закрылось)\s+'
        r'([A-Z][A-Za-z0-9&\'’.\- ]{1,60})',
        re.IGNORECASE
    ),

    # "Ресторан am.pm, ..."
    re.compile(
        r'(?:ресторан|кофейня|кафе|бар|корнер)\s+([A-Za-z0-9&\'’.\- ]{2,60})',
        re.IGNORECASE
    ),
]


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_address(text: str):
    text = clean_text(text)

    for i, pattern in enumerate(ADDRESS_PATTERNS, start=1):
        m = pattern.search(text)
        if m:
            address = m.group(1).strip(" .,")
            return address, f"address_rule_{i}"

    return None, None


def extract_name(text: str):
    text = clean_text(text)

    for i, pattern in enumerate(NAME_PATTERNS, start=1):
        m = pattern.search(text)
        if m:
            name = m.group(1).strip(" .,")
            # немного чистим ложные хвосты
            name = re.sub(r'\s+(в|на|по адресу)$', '', name, flags=re.IGNORECASE).strip()
            return name, f"name_rule_{i}"

    return None, None


def extract_metro_from_text(text: str):
    text = clean_text(text)
    hashtags = re.findall(r'#([а-яёa-z0-9_]+)', text, flags=re.IGNORECASE)
    if hashtags:
        # часто последний хэштег = метро, но это не всегда так
        return hashtags[-1]
    return None


results = []
for _, row in df.iterrows():
    text = row.get("text", "")

    venue_name, name_rule = extract_name(text)
    address, address_rule = extract_address(text)
    metro = extract_metro_from_text(text)

    confidence = "low"
    if venue_name and address:
        confidence = "high"
    elif venue_name or address:
        confidence = "medium"

    results.append({
        **row.to_dict(),
        "venue_name": venue_name,
        "address": address,
        "metro_from_text": metro,
        "name_rule": name_rule,
        "address_rule": address_rule,
        "extract_confidence": confidence,
    })

parsed_df = pd.DataFrame(results)
parsed_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"Saved: {OUTPUT_FILE}")
print(parsed_df[["venue_name", "address", "extract_confidence", "text"]].head(20).to_string(index=False))
import pandas as pd

# =========================
# 1. Пути к файлам
# =========================
INPUT_FILE = r"xxx"
OUTPUT_FILE = r"xxx"

# =========================
# 2. Название колонки
# =========================
REST_TYPE_COL = "rest_type"

# =========================
# 3. Словарь замены
# =========================
REST_TYPE_MAP = {
    "кафе": "Café",
    "ресторан": "Restaurant",
    "пиццерия": "Pizzeria",
    "бар": "Bar",
    "паб": "Pub",
    "клуб": "Club",
    "караокеклуб": "Karaoke club",
    "быстроепитание": "Fast-food outlet",
    "кофейня": "Coffee shop",
    "пекарня": "Bakery",
    "винотека": "Wine shop/bar",
    "кальянная": "Hookah lounge",
}

# =========================
# 4. Чтение файла
# =========================
df = pd.read_csv(INPUT_FILE)

# =========================
# 5. Проверки
# =========================
if REST_TYPE_COL not in df.columns:
    raise ValueError(f"Колонка '{REST_TYPE_COL}' не найдена в файле.")

# Приводим к строке и убираем пробелы по краям
df[REST_TYPE_COL] = df[REST_TYPE_COL].astype(str).str.strip()

# Сохраним исходные уникальные значения для контроля
before_values = sorted(df[REST_TYPE_COL].dropna().unique())
print("Уникальные значения ДО замены:")
for v in before_values:
    print(f" - {v}")

# =========================
# 6. Поиск незнакомых значений до замены
# =========================
unknown_before = sorted(set(df[REST_TYPE_COL].dropna().unique()) - set(REST_TYPE_MAP.keys()))
if unknown_before:
    print("\nВНИМАНИЕ: найдены значения, которых нет в словаре замены:")
    for v in unknown_before:
        print(f" - {v}")

# =========================
# 7. Замена значений
# =========================
df[REST_TYPE_COL] = df[REST_TYPE_COL].replace(REST_TYPE_MAP)

# =========================
# 8. Проверка после замены
# =========================
after_values = sorted(df[REST_TYPE_COL].dropna().unique())
print("\nУникальные значения ПОСЛЕ замены:")
for v in after_values:
    print(f" - {v}")

# Проверяем, остались ли русские / незамененные значения
expected_english_values = set(REST_TYPE_MAP.values())
unexpected_after = sorted(set(df[REST_TYPE_COL].dropna().unique()) - expected_english_values)

if unexpected_after:
    print("\nОШИБКА: после замены остались неожиданные значения:")
    for v in unexpected_after:
        print(f" - {v}")
    raise ValueError("Не все значения rest_type были успешно заменены.")
else:
    print("\nПроверка пройдена: все значения rest_type успешно заменены на английские.")

# =========================
# 9. Сохранение
# =========================
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nГотово. Файл сохранен: {OUTPUT_FILE}")
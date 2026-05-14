import pandas as pd
import folium
import html

# =========================
# Paths
# =========================

INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"
OUTPUT_FILE = "moscow_restaurants_map_fin_v2.html"

# =========================
# Columns
# =========================

LAT_COL = "latitude"
LON_COL = "longitude"
TARGET_COL = "target"
NAME_COL = "rest_name"

FIN_COLS = [
    "egrul_flg",
    "revenue_2025_m",
    "net_profit_2025_m",
    "arbitration_case_count",
    "inspection_count",
    "inn"
]

# =========================
# Read data
# =========================

df = pd.read_csv(INPUT_FILE)

# Убираем пробелы в названиях колонок
df.columns = df.columns.str.strip()

# Проверка обязательных колонок
required_cols = [LAT_COL, LON_COL, TARGET_COL]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Колонка '{col}' не найдена в файле.")

# Проверка финансовых колонок
missing_fin_cols = [col for col in FIN_COLS if col not in df.columns]

if missing_fin_cols:
    print(f"Предупреждение: в файле нет финансовых колонок: {missing_fin_cols}")

# =========================
# Normalize values
# =========================

# Координаты и target в числа
df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

# Финансовые колонки чистим как текст,
# чтобы не терять ИНН и нормально обрабатывать '-', пробелы, NaN
for col in FIN_COLS:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\xa0", "", regex=False)
            .str.strip()
        )

# Убираем строки без координат или target
df = df.dropna(subset=[LAT_COL, LON_COL, TARGET_COL]).copy()

# =========================
# Helper functions
# =========================

def safe_value(row, col):
    """
    Безопасный вывод значения в popup.
    Возвращает N/A для пустых значений, '-', nan и отсутствующих колонок.
    """
    if col not in row.index:
        return "COLUMN_NOT_FOUND"

    value = row[col]

    if pd.isna(value):
        return "N/A"

    value_str = str(value).replace("\xa0", "").strip()

    if value_str in ["", "-", "nan", "NaN", "None", "NONE", "null", "NULL"]:
        return "N/A"

    return html.escape(value_str)


def format_money(row, col):
    """
    Форматирование денежных показателей.
    Значения предполагаются в млн рублей.
    """
    value = safe_value(row, col)

    if value in ["N/A", "COLUMN_NOT_FOUND"]:
        return value

    try:
        num = float(str(value).replace(",", "."))
        return f"{num:.2f}"
    except ValueError:
        return value


def format_int(row, col):
    """
    Форматирование целочисленных показателей:
    arbitration_cases, inspections, egrul_flg.
    """
    value = safe_value(row, col)

    if value in ["N/A", "COLUMN_NOT_FOUND"]:
        return value

    try:
        num = float(str(value).replace(",", "."))
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return value


# =========================
# Optional debug check
# =========================

# Можно раскомментировать и проверить конкретный ИНН
# CHECK_INN = "9705148806"
# print(
#     df.loc[
#         df["inn"].astype(str).str.contains(CHECK_INN, na=False),
#         [
#             NAME_COL,
#             "egrul_flg",
#             "revenue_2025_m",
#             "net_profit_2025_m",
#             "arbitration_cases",
#             "inspections",
#             "inn"
#         ]
#     ].to_string()
# )

# =========================
# Map
# =========================

center_lat = df[LAT_COL].mean()
center_lon = df[LON_COL].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="OpenStreetMap"
)

for _, row in df.iterrows():
    target = int(row[TARGET_COL])
    color = "green" if target == 0 else "red"

    rest_name = safe_value(row, NAME_COL)

    tooltip_html = f"""
    <b>{rest_name}</b><br>
    target: {target}
    """

    popup_html = f"""
    <div style="font-family: Arial; font-size: 13px; width: 300px;">
        <b>{rest_name}</b><br>
        <b>Target:</b> {target}<br><br>

        <b>Financial and legal indicators</b><br>
        <b>egrul_flg:</b> {format_int(row, "egrul_flg")}<br>
        <b>revenue_2025_m:</b> {format_money(row, "revenue_2025_m")} mln RUB<br>
        <b>net_profit_2025_m:</b> {format_money(row, "net_profit_2025_m")} mln RUB<br>
        <b>arbitration_cases:</b> {format_int(row, "arbitration_case_count")}<br>
        <b>inspections:</b> {format_int(row, "inspection_count")}<br>
        <b>inn:</b> {safe_value(row, "inn")}
    </div>
    """

    folium.CircleMarker(
        location=[row[LAT_COL], row[LON_COL]],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
        popup=folium.Popup(popup_html, max_width=350)
    ).add_to(m)

# =========================
# Save
# =========================

m.save(OUTPUT_FILE)

print(f"Карта сохранена в файл: {OUTPUT_FILE}")
print(f"Количество точек на карте: {len(df)}")

# =========================
# Quick coverage check
# =========================

print("\nПокрытие финансовых колонок:")

for col in FIN_COLS:
    if col in df.columns:
        valid_count = (
            df[col]
            .astype(str)
            .str.strip()
            .replace(["", "-", "nan", "NaN", "None", "NULL", "null"], pd.NA)
            .notna()
            .sum()
        )
        print(f"{col}: {valid_count} заполнено из {len(df)}")
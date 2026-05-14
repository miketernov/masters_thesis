import pandas as pd

INPUT_FILE = r"xxx"
OUTPUT_FILE = r"xxx"

df = pd.read_csv(INPUT_FILE)

rating_cols = [
    "comp_rating_avg_100m",
    "comp_rating_avg_200m",
    "comp_rating_avg_300m",
]

count_cols = ["comp_100m", "comp_200m", "comp_300m"]

# --- 1. Чистим данные ---
df[rating_cols] = df[rating_cols].replace("", pd.NA)

# нормализуем тип
df["rest_type_norm"] = (
    df["rest_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "", regex=False)
)

# --- 2. Флаги отсутствия конкурентов ---
df["flag_no_comp_100m"] = (df["comp_100m"] == 0).astype(int)
df["flag_no_comp_200m"] = (df["comp_200m"] == 0).astype(int)

# --- 3. Считаем средние рейтинги ---
type_rating_means = (
    df.groupby("rest_type_norm")[rating_cols]
    .mean()
)

global_rating_means = df[rating_cols].mean()

# --- 4. Импутация рейтингов ---
filled_count = 0

for i, row in df.iterrows():
    rest_type = row["rest_type_norm"]

    for col in rating_cols:
        if pd.isna(row[col]):

            if rest_type in type_rating_means.index and pd.notna(type_rating_means.loc[rest_type, col]):
                value = type_rating_means.loc[rest_type, col]
            else:
                value = global_rating_means[col]

            df.at[i, col] = round(float(value), 4)
            filled_count += 1

# --- 5. Очистка ---
df.drop(columns=["rest_type_norm"], inplace=True)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"Готово. Заполнено рейтингов: {filled_count}")
print(f"Файл сохранен: {OUTPUT_FILE}")
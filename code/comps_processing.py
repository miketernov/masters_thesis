import pandas as pd

INPUT_FILE = r"xxx"
OUTPUT_FILE = r"xxx"

df = pd.read_csv(INPUT_FILE)

target_cols = [
    "comp_100m",
    "comp_200m",
    "comp_300m",
    "comp_rating_avg_100m",
    "comp_rating_avg_200m",
    "comp_rating_avg_300m",
]

count_cols = ["comp_100m", "comp_200m", "comp_300m"]
rating_cols = [
    "comp_rating_avg_100m",
    "comp_rating_avg_200m",
    "comp_rating_avg_300m",
]

# на случай если где-то пустые строки, а не NaN
df[target_cols] = df[target_cols].replace("", pd.NA)

# нормализуем rest_type, чтобы группировка была чище
df["rest_type_norm"] = (
    df["rest_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "", regex=False)
)

# строки, где все 6 полей пустые
mask_all_empty = df[target_cols].isna().all(axis=1)

# считаем средние по типу ТОЛЬКО по строкам, где уже есть данные
df_non_empty = df[~mask_all_empty].copy()

type_means = (
    df_non_empty
    .groupby("rest_type_norm")[target_cols]
    .mean()
)

# глобальные средние на случай, если для типа нет данных
global_means = df_non_empty[target_cols].mean()

filled_count = 0

for i, row in df[mask_all_empty].iterrows():
    rest_type_norm = row["rest_type_norm"]

    for col in target_cols:
        if rest_type_norm in type_means.index and pd.notna(type_means.loc[rest_type_norm, col]):
            value = type_means.loc[rest_type_norm, col]
        else:
            value = global_means[col]

        # счетчики округляем до целого
        if col in count_cols and pd.notna(value):
            value = int(round(value))

        # рейтинги округляем до 4 знаков
        if col in rating_cols and pd.notna(value):
            value = round(float(value), 4)

        df.at[i, col] = value

    filled_count += 1

# убираем служебную колонку
df.drop(columns=["rest_type_norm"], inplace=True)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Готово. Заполнено строк: {filled_count}")
print(f"Файл сохранен: {OUTPUT_FILE}")
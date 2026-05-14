import pandas as pd

INPUT_FILE = r"xxx"
OUTPUT_FILE = r"xxx"

df = pd.read_csv(INPUT_FILE)

# --- приводим rating_count к числу ---
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

# --- создаем новую колонку ---
df["review_count"] = (df["rating_count"] / 1.5).round()

# если хочешь целые значения (с поддержкой NaN)
df["review_count"] = df["review_count"].astype("Int64")

# сохраняем
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Готово")
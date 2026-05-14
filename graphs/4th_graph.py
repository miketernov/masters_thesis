import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Путь к файлу
# =========================
INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"

# =========================
# 2. Названия колонок
# =========================
TARGET_COL = "target"       # 0 = survived >1 year, 1 = closed <1 year
RATING_COL = "avg_rating"

# =========================
# 3. Чтение файла
# =========================
df = pd.read_csv(INPUT_FILE)

# Проверка наличия колонок
for col in [TARGET_COL, RATING_COL]:
    if col not in df.columns:
        raise ValueError(f"Колонка '{col}' не найдена в файле.")

# =========================
# 4. Подготовка данных
# =========================
df = df[[TARGET_COL, RATING_COL]].copy()
df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL, RATING_COL])

# Данные по классам
survived = df[df[TARGET_COL] == 0][RATING_COL]
closed = df[df[TARGET_COL] == 1][RATING_COL]

# =========================
# 5. Построение boxplot
# =========================
plt.figure(figsize=(8, 6))
plt.boxplot(
    [survived, closed],
    labels=["0 - survived >1 year", "1 - closed <1 year"],
    patch_artist=False
)

plt.title("Average rating by target class")
plt.xlabel("Class")
plt.ylabel("Average rating")
plt.tight_layout()
plt.show()

# =========================
# 6. Краткая статистика
# =========================
summary = df.groupby(TARGET_COL)[RATING_COL].agg(["count", "mean", "median", "std", "min", "max"])
print(summary)
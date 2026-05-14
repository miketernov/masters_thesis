import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Путь к файлу
# =========================
INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"

# =========================
# 2. Название колонки
# =========================
TARGET_COL = "target"

# =========================
# 3. Чтение файла
# =========================
df = pd.read_csv(INPUT_FILE)

# Проверка наличия колонки
if TARGET_COL not in df.columns:
    raise ValueError(f"Колонка '{TARGET_COL}' не найдена в файле.")

# =========================
# 4. Подсчет значений
# =========================
counts = df[TARGET_COL].value_counts(dropna=False).sort_index()

# Оставляем только 0 и 1, если нужно
counts = counts.reindex([0, 1], fill_value=0)

# Подписи для оси X
labels = ["0 - survived >1 year", "1 - closed <1 year"]

# =========================
# 5. Построение графика
# =========================
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, counts.values)

# Подписи над столбцами
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.title("Distribution of restaurants by survival_1y")
plt.xlabel("Class")
plt.ylabel("Number of restaurants")
plt.tight_layout()
plt.show()
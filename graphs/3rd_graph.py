import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Путь к файлу
# =========================
INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"

# =========================
# 2. Названия колонок
# =========================
TYPE_COL = "rest_type"
TARGET_COL = "target"   # 1 = closed <1 year, 0 = survived >1 year

# =========================
# 3. Чтение файла
# =========================
df = pd.read_csv(INPUT_FILE)

# Проверки
for col in [TYPE_COL, TARGET_COL]:
    if col not in df.columns:
        raise ValueError(f"Колонка '{col}' не найдена в файле.")

# =========================
# 4. Подготовка данных
# =========================
df[TYPE_COL] = df[TYPE_COL].astype(str).str.strip()

# closure rate = доля target=1 внутри каждого типа
summary = (
    df.groupby(TYPE_COL)[TARGET_COL]
    .agg(
        total_count="count",
        closed_count="sum"
    )
    .reset_index()
)

summary["closure_rate"] = summary["closed_count"] / summary["total_count"]

# Можно отсортировать по доле закрытия
summary = summary.sort_values("closure_rate", ascending=False)

# =========================
# 5. Построение графика
# =========================
plt.figure(figsize=(12, 6))
bars = plt.bar(summary[TYPE_COL], summary["closure_rate"])

# Подписи над столбцами: процент + count
for bar, rate, closed, total in zip(
    bars,
    summary["closure_rate"],
    summary["closed_count"],
    summary["total_count"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{rate:.1%}\n({int(closed)}/{int(total)})",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.title("Early closure rate by restaurant type")
plt.xlabel("Restaurant type")
plt.ylabel("Closure rate")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, summary["closure_rate"].max() * 1.15)
plt.tight_layout()
plt.show()

# =========================
# 6. Печать таблицы для проверки
# =========================
print(summary)
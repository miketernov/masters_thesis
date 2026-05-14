import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Путь к файлу
# =========================
INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"

# =========================
# 2. Колонки
# =========================
TARGET_COL = "target"
SENTIMENT_SHARE_COLS = ["positive_share", "neutral_share", "negative_share"]
AVG_SENTIMENT_COL = "avg_sentiment_score"

# =========================
# 3. Чтение файла
# =========================
df = pd.read_csv(INPUT_FILE)

required_cols = [TARGET_COL] + SENTIMENT_SHARE_COLS + [AVG_SENTIMENT_COL]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Колонка '{col}' не найдена в файле.")

# Приводим к numeric
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 4. Средние доли sentiment по target
# =========================
summary = (
    df.groupby(TARGET_COL)[SENTIMENT_SHARE_COLS]
    .mean()
    .reindex([0, 1])
)

labels = ["0 - survived >1 year", "1 - closed <1 year"]
x = np.arange(len(labels))
width = 0.22

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width, summary["positive_share"], width, label="positive_share")
bars2 = plt.bar(x, summary["neutral_share"], width, label="neutral_share")
bars3 = plt.bar(x + width, summary["negative_share"], width, label="negative_share")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(x, labels)
plt.ylabel("Mean share")
plt.xlabel("Class")
plt.title("Mean sentiment shares by target class")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 5. Boxplot для avg_sentiment_score
# =========================
plot_df = df[[TARGET_COL, AVG_SENTIMENT_COL]].dropna()

survived = plot_df[plot_df[TARGET_COL] == 0][AVG_SENTIMENT_COL]
closed = plot_df[plot_df[TARGET_COL] == 1][AVG_SENTIMENT_COL]

plt.figure(figsize=(8, 6))
plt.boxplot(
    [survived, closed],
    labels=["0 - survived >1 year", "1 - closed <1 year"]
)
plt.title("Average sentiment score by target class")
plt.xlabel("Class")
plt.ylabel("Average sentiment score")
plt.tight_layout()
plt.show()

# =========================
# 6. Таблица со средними
# =========================
print("\nMean sentiment shares by target:")
print(summary)

print("\nSummary for avg_sentiment_score:")
print(
    plot_df.groupby(TARGET_COL)[AVG_SENTIMENT_COL]
    .agg(["count", "mean", "median", "std", "min", "max"])
)
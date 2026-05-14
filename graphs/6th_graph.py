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

COMPLAINT_COLS = [
    "service_complaint_share",
    "food_complaint_share",
    "price_complaint_share"
]

AMONG_NEGATIVE_COLS = [
    "service_among_negative_share",
    "food_among_negative_share",
    "price_among_negative_share"
]

# =========================
# 3. Чтение файла
# =========================
df = pd.read_csv(INPUT_FILE)

required_cols = [TARGET_COL] + COMPLAINT_COLS + AMONG_NEGATIVE_COLS
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Колонка '{col}' не найдена в файле.")

for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

labels = ["0 - survived >1 year", "1 - closed <1 year"]
x = np.arange(len(labels))
width = 0.22

# =========================
# 4. Complaint shares by target
# =========================
summary_complaints = (
    df.groupby(TARGET_COL)[COMPLAINT_COLS]
    .mean()
    .reindex([0, 1])
)

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width, summary_complaints["service_complaint_share"], width, label="service_complaint_share")
bars2 = plt.bar(x, summary_complaints["food_complaint_share"], width, label="food_complaint_share")
bars3 = plt.bar(x + width, summary_complaints["price_complaint_share"], width, label="price_complaint_share")

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
plt.title("Mean complaint shares by target class")
plt.legend()
plt.tight_layout()
plt.show()

print("\nMean complaint shares by target:")
print(summary_complaints)

# =========================
# 5. Complaint shares among negative reviews
# =========================
summary_among_negative = (
    df.groupby(TARGET_COL)[AMONG_NEGATIVE_COLS]
    .mean()
    .reindex([0, 1])
)

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width, summary_among_negative["service_among_negative_share"], width, label="service_among_negative_share")
bars2 = plt.bar(x, summary_among_negative["food_among_negative_share"], width, label="food_among_negative_share")
bars3 = plt.bar(x + width, summary_among_negative["price_among_negative_share"], width, label="price_among_negative_share")

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
plt.title("Complaint shares among negative reviews by target class")
plt.legend()
plt.tight_layout()
plt.show()

print("\nMean complaint shares among negative reviews by target:")
print(summary_among_negative)
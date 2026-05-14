import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"

corr_cols = [
    "target",
    "avg_bill",
    "avg_rating",
    "rating_count",
    "review_count",
    "metro_distance",
    "center_distance",
    "comp_100m",
    "comp_200m",
    "comp_300m",
    "comp_rating_avg_100m",
    "comp_rating_avg_200m",
    "comp_rating_avg_300m",
    "avg_review_len_chars",
    "avg_review_len_words",
    "positive_share",
    "neutral_share",
    "negative_share",
    "avg_sentiment_score",
    "service_complaint_share",
    "food_complaint_share",
    "price_complaint_share",
    "negative_service_share",
    "negative_food_share",
    "negative_price_share",
    "service_among_negative_share",
    "food_among_negative_share",
    "price_among_negative_share"
]

df = pd.read_csv(INPUT_FILE)

missing = [col for col in corr_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in corr_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

corr_df = df[corr_cols].copy()
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(11, 8))

im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(corr_cols)))
ax.set_yticks(np.arange(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="right")
ax.set_yticklabels(corr_cols)

# Подписи значений внутри ячеек
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(
            j, i, f"{corr.values[i, j]:.2f}",
            ha="center", va="center", fontsize=8
        )

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Correlation coefficient")

plt.title("Correlation matrix for key numerical variables")
plt.tight_layout()
plt.show()

print(corr)
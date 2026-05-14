import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

# На случай если в данных есть "-" или пустые строки
def is_filled(series):
    return (
        series.notna()
        & (series.astype(str).str.strip() != "")
        & (series.astype(str).str.strip() != "-")
        & (series.astype(str).str.lower().str.strip() != "nan")
    )

# -----------------------------
# 1. SPARK coverage bar chart
# -----------------------------

spark_filled = is_filled(df["spark_link"])

spark_counts = pd.Series({
    "Matched with SPARK": spark_filled.sum(),
    "Not matched": (~spark_filled).sum()
})

plt.figure(figsize=(7, 5))
spark_counts.plot(kind="bar")

plt.title("SPARK Matching Coverage")
plt.ylabel("Number of restaurants")
plt.xticks(rotation=0)

for i, value in enumerate(spark_counts.values):
    plt.text(i, value + 5, str(value), ha="center")

plt.tight_layout()
plt.savefig("spark_matching_coverage.png", dpi=300, bbox_inches="tight")
plt.show()

print(spark_counts)
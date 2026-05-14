import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

# Чистим profit
df["net_profit_2025_m"] = (
    df["net_profit_2025_m"]
    .astype(str)
    .str.replace("\xa0", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
)

df["net_profit_2025_m"] = pd.to_numeric(df["net_profit_2025_m"], errors="coerce")

# Берем только строки, где есть spark_link
spark_filled = (
    df["spark_link"].notna()
    & (df["spark_link"].astype(str).str.strip() != "")
    & (df["spark_link"].astype(str).str.strip() != "-")
)

df_spark = df[spark_filled].copy()

profit_counts = pd.Series({
    "Net profit > 0": (df_spark["net_profit_2025_m"] > 0).sum(),
    "Net profit = 0": (df_spark["net_profit_2025_m"] == 0).sum(),
    "Net profit < 0": (df_spark["net_profit_2025_m"] < 0).sum(),
    "No profit data": df_spark["net_profit_2025_m"].isna().sum()
})

plt.figure(figsize=(8, 5))
profit_counts.plot(kind="bar")

plt.title("Net Profit Distribution among SPARK-Matched Restaurants")
plt.ylabel("Number of restaurants")
plt.xticks(rotation=0)

for i, value in enumerate(profit_counts.values):
    plt.text(i, value + 2, str(value), ha="center")

plt.tight_layout()
plt.savefig("net_profit_distribution_spark_matched.png", dpi=300, bbox_inches="tight")
plt.show()

print(profit_counts)
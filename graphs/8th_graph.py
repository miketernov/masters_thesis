import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_4.csv"

TARGET_COL = "target"
QUALITY_COL = "comp_rating_avg_100m"

df = pd.read_csv(INPUT_FILE)

for col in [TARGET_COL, QUALITY_COL]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

df[QUALITY_COL] = pd.to_numeric(df[QUALITY_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL, QUALITY_COL])

survived = df[df[TARGET_COL] == 0][QUALITY_COL]
closed = df[df[TARGET_COL] == 1][QUALITY_COL]

plt.figure(figsize=(8, 6))
plt.boxplot(
    [survived, closed],
    labels=["0 - survived >1 year", "1 - closed <1 year"]
)
plt.title("Competitor quality indicator (comp_rating_avg_100m) by target class")
plt.xlabel("Class")
plt.ylabel("Average competitor rating within 100m")
plt.tight_layout()
plt.show()

print(df.groupby(TARGET_COL)[QUALITY_COL].agg(["count", "mean", "median", "std", "min", "max"]))
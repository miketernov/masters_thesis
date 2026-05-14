import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_4.csv"

TARGET_COL = "target"      # 0 = survived >1 year, 1 = closed <1 year
COMP_COL = "comp_100m"

df = pd.read_csv(INPUT_FILE)

for col in [TARGET_COL, COMP_COL]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

df[COMP_COL] = pd.to_numeric(df[COMP_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL, COMP_COL])

survived = df[df[TARGET_COL] == 0][COMP_COL]
closed = df[df[TARGET_COL] == 1][COMP_COL]

plt.figure(figsize=(8, 6))
plt.boxplot(
    [survived, closed],
    labels=["0 - survived >1 year", "1 - closed <1 year"]
)
plt.title("Competition indicator (comp_100m) by target class")
plt.xlabel("Class")
plt.ylabel("Number of competitors within 100m")
plt.tight_layout()
plt.show()

print(df.groupby(TARGET_COL)[COMP_COL].agg(["count", "mean", "median", "std", "min", "max"]))
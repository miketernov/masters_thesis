import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_4.csv"

TARGET_COL = "target"
LOCATION_COLS = ["metro_distance", "center_distance"]

df = pd.read_csv(INPUT_FILE)

for col in [TARGET_COL] + LOCATION_COLS:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")

for col in LOCATION_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for loc_col in LOCATION_COLS:
    temp = df.dropna(subset=[TARGET_COL, loc_col])

    survived = temp[temp[TARGET_COL] == 0][loc_col]
    closed = temp[temp[TARGET_COL] == 1][loc_col]

    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [survived, closed],
        labels=["0 - survived >1 year", "1 - closed <1 year"]
    )
    plt.title(f"{loc_col} by target class")
    plt.xlabel("Class")
    plt.ylabel(loc_col)
    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for {loc_col}:")
    print(temp.groupby(TARGET_COL)[loc_col].agg(["count", "mean", "median", "std", "min", "max"]))
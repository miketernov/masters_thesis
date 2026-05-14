import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"
TYPE_COL = "rest_type"

ordered_types = [
    "Café",
    "Restaurant",
    "Pizzeria",
    "Bar",
    "Pub",
    "Club",
    "Karaoke club",
    "Fast-food outlet",
    "Coffee shop",
    "Bakery",
    "Wine shop/bar",
    "Hookah lounge",
]

df = pd.read_csv(INPUT_FILE)

if TYPE_COL not in df.columns:
    raise ValueError(f"Колонка '{TYPE_COL}' не найдена в файле.")

type_counts = (
    df[TYPE_COL]
    .astype(str)
    .str.strip()
    .value_counts()
    .reindex(ordered_types, fill_value=0)
)

plt.figure(figsize=(12, 6))
bars = plt.bar(type_counts.index, type_counts.values)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.title("Distribution of establishments by restaurant type")
plt.xlabel("Restaurant type")
plt.ylabel("Number of establishments")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
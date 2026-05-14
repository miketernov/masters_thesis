import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\restaurants_with_review_features_v_5.csv"
OUTPUT_CORR_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\full_correlation_matrix.csv"
OUTPUT_STRONG_CORR_FILE = r"C:\Users\test_\PycharmProjects\DIPLOMA_MAGA_TEST\strong_correlation_pairs.csv"

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

# =========================
# 2. TRY TO CONVERT COLUMNS TO NUMERIC
# =========================
df_for_corr = df.copy()

for col in df_for_corr.columns:
    if df_for_corr[col].dtype == bool:
        df_for_corr[col] = df_for_corr[col].astype(int)
    elif df_for_corr[col].dtype == "object":
        converted = pd.to_numeric(df_for_corr[col], errors="coerce")
        # заменяем на numeric только если есть хотя бы одно числовое значение
        if converted.notna().sum() > 0:
            df_for_corr[col] = converted

# =========================
# 3. SELECT ONLY NUMERIC COLUMNS
# =========================
numeric_df = df_for_corr.select_dtypes(include=["number"]).copy()

if numeric_df.shape[1] == 0:
    raise ValueError("В файле не найдено числовых колонок для корреляции.")

print("Числовые признаки для корреляции:")
print(numeric_df.columns.tolist())

# =========================
# 4. CORRELATION MATRIX
# =========================
corr = numeric_df.corr()

# сохраняем полную матрицу корреляции
corr.to_csv(OUTPUT_CORR_FILE, encoding="utf-8-sig")

# =========================
# 5. PLOT FULL HEATMAP
# =========================
n = len(corr.columns)

fig_w = max(12, n * 0.6)
fig_h = max(10, n * 0.6)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)

# Подписываем значения только если признаков не слишком много
if n <= 20:
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Correlation coefficient")

plt.title("Full correlation matrix for all numerical variables")
plt.tight_layout()
plt.show()

# =========================
# 6. PRINT FULL MATRIX
# =========================
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 2000)

print("\nПолная матрица корреляции:")
print(corr)

# =========================
# 7. FIND STRONG CORRELATIONS
# =========================
threshold = 0.7

strong_pairs = []
cols = corr.columns.tolist()

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        val = corr.iloc[i, j]
        if abs(val) >= threshold:
            strong_pairs.append({
                "feature_1": cols[i],
                "feature_2": cols[j],
                "correlation": val
            })

strong_df = pd.DataFrame(strong_pairs)

if not strong_df.empty:
    strong_df = strong_df.sort_values(
        by="correlation",
        key=lambda s: s.abs(),
        ascending=False
    )
    strong_df.to_csv(OUTPUT_STRONG_CORR_FILE, index=False, encoding="utf-8-sig")

    print(f"\nПары признаков с |corr| >= {threshold}:")
    print(strong_df.to_string(index=False))

    print(f"\nФайл с сильными корреляциями сохранен:")
    print(OUTPUT_STRONG_CORR_FILE)
else:
    print(f"\nНет пар признаков с |corr| >= {threshold}")

print(f"\nПолная матрица корреляции сохранена:")
print(OUTPUT_CORR_FILE)
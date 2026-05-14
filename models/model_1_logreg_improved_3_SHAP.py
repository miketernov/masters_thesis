import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# =========================
# 1. PATHS
# =========================
INPUT_FILE = r"xxx"
OUTPUT_DIR = r"xxx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. SETTINGS
# =========================
TARGET_COL = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_WEIGHT = None
RUN_SHAP = True   # если SHAP не нужен, поставь False

# =========================
# 3. LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

if TARGET_COL not in df.columns:
    raise ValueError(f"Column '{TARGET_COL}' not found in dataset.")

print("Dataset shape:", df.shape)
print("Target distribution:")
print(df[TARGET_COL].value_counts(dropna=False))
print(df[TARGET_COL].value_counts(normalize=True, dropna=False))

# =========================
# 4. FEATURE ENGINEERING
# =========================
required_for_engineering = [
    "avg_rating",
    "comp_rating_avg_300m",
    "reviews_count_collected",
    "service_complaint_share",
    "food_complaint_share",
    "price_complaint_share",
    "avg_bill",
]

missing_for_engineering = [col for col in required_for_engineering if col not in df.columns]
if missing_for_engineering:
    raise ValueError(
        f"Missing columns required for feature engineering: {missing_for_engineering}"
    )

# 1) Restaurant rating relative to competitors
df["rating_vs_comp"] = df["avg_rating"] - df["comp_rating_avg_300m"]

# 2) Low number of collected reviews
df["low_reviews"] = (df["reviews_count_collected"] < 20).astype(int)

# 3) Total complaint share
df["total_complaint_share"] = (
    df["service_complaint_share"].fillna(0)
    + df["food_complaint_share"].fillna(0)
    + df["price_complaint_share"].fillna(0)
)

# 4) Log average bill
df["log_avg_bill"] = np.log1p(df["avg_bill"])

# =========================
# 5. EXCLUDE NON-MODEL / REDUNDANT COLUMNS
# =========================
excluded_cols = {
    TARGET_COL,

    # ID / raw text / technical
    "restaurant_id",
    "date",
    "rest_name",
    "full_address",
    "latitude",
    "longitude",
    "metro_station",

    # excluded as in your version
    "rating_count",
    "review_count",

    # replaced with engineered feature
    "avg_bill",

    # sentiment multicollinearity
    "positive_share",
    "negative_share",

    # duplicate / near duplicate
    "avg_review_len_chars",

    # correlated competition ratings
    "comp_rating_avg_200m",

    # almost zero variance
    "flag_no_comp_200m",
}

feature_cols = [col for col in df.columns if col not in excluded_cols]

if not feature_cols:
    raise ValueError("No feature columns left after exclusions.")

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

print("\nNumber of features before preprocessing:", X.shape[1])

# =========================
# 6. IDENTIFY FEATURE TYPES
# =========================
numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [col for col in X.columns if col not in numeric_features]

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# =========================
# 7. PREPROCESSING
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# =========================
# 8. MODEL
# =========================
model = LogisticRegression(
    max_iter=3000,
    solver="liblinear",
    class_weight=CLASS_WEIGHT,
    random_state=RANDOM_STATE,
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model),
])

# =========================
# 9. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True).sort_index())

# =========================
# 10. FIT
# =========================
clf.fit(X_train, y_train)

# =========================
# 11. PREDICT
# =========================
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# =========================
# 12. METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

metrics_df = pd.DataFrame({
    "Model": ["Logistic Regression"],
    "Accuracy": [acc],
    "Precision": [prec],
    "Recall": [rec],
    "F1": [f1],
    "ROC_AUC": [roc_auc],
})

print("\n=== Logistic Regression Metrics ===")
print(metrics_df.to_string(index=False))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

metrics_path = os.path.join(OUTPUT_DIR, "logreg_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

# =========================
# 13. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n=== Confusion Matrix ===")
print(cm)
print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="viridis", values_format="d")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logreg_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 14. FEATURE NAMES AFTER PREPROCESSING
# =========================
preprocessor_fitted = clf.named_steps["preprocessor"]
model_fitted = clf.named_steps["model"]

feature_names = preprocessor_fitted.get_feature_names_out().tolist()
feature_names = [
    name.replace("num__", "").replace("cat__", "")
    for name in feature_names
]

coefficients = model_fitted.coef_[0]

# =========================
# 15. COEFFICIENT TABLE
# =========================
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "abs_coefficient": np.abs(coefficients),
    "odds_ratio": np.exp(coefficients),
})

coef_df_sorted_abs = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
coef_df_sorted_pos = coef_df.sort_values("coefficient", ascending=False).reset_index(drop=True)
coef_df_sorted_neg = coef_df.sort_values("coefficient", ascending=True).reset_index(drop=True)

print("\n=== Top 20 features by absolute coefficient ===")
print(coef_df_sorted_abs.head(20).to_string(index=False))

print("\n=== Top 15 positive coefficients (increase probability of target=1) ===")
print(coef_df_sorted_pos.head(15).to_string(index=False))

print("\n=== Top 15 negative coefficients (decrease probability of target=1) ===")
print(coef_df_sorted_neg.head(15).to_string(index=False))

coef_path = os.path.join(OUTPUT_DIR, "logreg_coefficients_full.csv")
coef_df_sorted_abs.to_csv(coef_path, index=False)

# =========================
# 16. TOP COEFFICIENTS PLOT
# =========================
top_n = 15
plot_df = coef_df_sorted_abs.head(top_n).copy()
plot_df = plot_df.sort_values("coefficient")

plt.figure(figsize=(10, 7))
plt.barh(plot_df["feature"], plot_df["coefficient"])
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.title("Top Logistic Regression Coefficients")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logreg_top_coefficients.png"), dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 17. TOP POSITIVE / NEGATIVE TABLES
# =========================
top_positive = coef_df_sorted_pos.head(10).copy()
top_negative = coef_df_sorted_neg.head(10).copy()

top_positive.to_csv(os.path.join(OUTPUT_DIR, "logreg_top_positive_coefficients.csv"), index=False)
top_negative.to_csv(os.path.join(OUTPUT_DIR, "logreg_top_negative_coefficients.csv"), index=False)

# =========================
# 18. OPTIONAL: SHAP
# =========================
if RUN_SHAP:
    try:
        import shap

        print("\n=== SHAP calculation started ===")

        X_train_transformed = preprocessor_fitted.transform(X_train)
        X_test_transformed = preprocessor_fitted.transform(X_test)

        # convert sparse -> dense if needed
        if hasattr(X_train_transformed, "toarray"):
            X_train_transformed = X_train_transformed.toarray()
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()

        # optional: use only a sample of train for explainer background if dataset is large
        if len(X_train_transformed) > 300:
            rng = np.random.default_rng(RANDOM_STATE)
            sample_idx = rng.choice(len(X_train_transformed), size=300, replace=False)
            X_train_background = X_train_transformed[sample_idx]
        else:
            X_train_background = X_train_transformed

        explainer = shap.Explainer(
            model_fitted,
            X_train_background,
            feature_names=feature_names,
        )

        shap_values = explainer(X_test_transformed)

        # SHAP importance table
        shap_importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        print("\n=== Top 15 SHAP features ===")
        print(shap_importance.head(15).to_string(index=False))

        shap_importance.to_csv(
            os.path.join(OUTPUT_DIR, "logreg_shap_importance.csv"),
            index=False
        )

        # SHAP bar plot
        plt.figure()
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title("SHAP Feature Importance - Logistic Regression")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "logreg_shap_bar.png"), dpi=300, bbox_inches="tight")
        plt.show()

        # SHAP beeswarm plot
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.title("SHAP Summary Plot - Logistic Regression")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "logreg_shap_beeswarm.png"), dpi=300, bbox_inches="tight")
        plt.show()

        print("\nSHAP plots and tables were successfully saved.")

    except ImportError:
        print("\nSHAP is not installed.")
        print("Install it with: pip install shap")
    except Exception as e:
        print(f"\nSHAP could not be calculated: {e}")

# =========================
# 19. SAVE SHORT INTERPRETATION TABLE
# =========================
interpretation_df = coef_df_sorted_abs.copy()

def interpret_direction(x):
    if x > 0:
        return "increases probability of target=1"
    elif x < 0:
        return "decreases probability of target=1"
    return "no effect"

interpretation_df["direction"] = interpretation_df["coefficient"].apply(interpret_direction)
interpretation_df["odds_ratio"] = interpretation_df["odds_ratio"].round(4)
interpretation_df["coefficient"] = interpretation_df["coefficient"].round(4)
interpretation_df["abs_coefficient"] = interpretation_df["abs_coefficient"].round(4)

interpretation_df.to_csv(
    os.path.join(OUTPUT_DIR, "logreg_interpretation_table.csv"),
    index=False
)

print("\nAll main outputs are saved to:")
print(OUTPUT_DIR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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
from catboost import CatBoostClassifier

# =========================
# 1. PATHS
# =========================
INPUT_FILE = r"xxx"

# =========================
# 2. SETTINGS
# =========================
TARGET_COL = "target"
TEST_SIZE = 0.2
VALID_SIZE = 0.2   # доля от train_part
RANDOM_STATE = 42

# =========================
# 3. LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

if TARGET_COL not in df.columns:
    raise ValueError(f"Column '{TARGET_COL}' not found in dataset.")

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

# restaurant rating relative to local competitors
df["rating_vs_comp"] = df["avg_rating"] - df["comp_rating_avg_300m"]

# insufficient review volume flag
df["low_reviews"] = (df["reviews_count_collected"] < 20).astype(int)

# overall complaint intensity
df["total_complaint_share"] = (
    df["service_complaint_share"].fillna(0)
    + df["food_complaint_share"].fillna(0)
    + df["price_complaint_share"].fillna(0)
)

# log-transformed average bill
df["log_avg_bill"] = np.log1p(df["avg_bill"])

# =========================
# 5. EXCLUDE NON-MODEL / REDUNDANT COLUMNS
# =========================
excluded_cols = {
    # target
    TARGET_COL,

    # technical / identifiers / raw text
    "restaurant_id",
    "date",
    "rest_name",
    "full_address",
    "latitude",
    "longitude",
    "metro_station",

    # excluded for now, can be tested later separately
    "rating_count",
    "review_count",

    # replaced by engineered feature
    "avg_bill",

    # sentiment multicollinearity
    "positive_share",
    #"neutral_share",
    "negative_share",
    #"avg_sentiment_score",

    # near-duplicate length feature
    "avg_review_len_chars",

    # highly correlated competition counts
    #"comp_100m",
    #"comp_200m",

    # highly correlated competition rating features
    #"comp_rating_avg_100m",
    #"comp_rating_avg_200m",

    # highly correlated complaint-derived features
    #"negative_service_share",
    #"negative_food_share",
    #"negative_price_share",

    # almost zero variance
    "flag_no_comp_200m",
}

feature_cols = [col for col in df.columns if col not in excluded_cols]

if not feature_cols:
    raise ValueError("No feature columns left after exclusions.")

X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int).copy()

print("All model features:")
print(X.columns.tolist())

# =========================
# 6. IDENTIFY CATEGORICAL FEATURES
# =========================
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

# =========================
# 7. TRAIN / VALID / TEST SPLIT
# =========================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VALID_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

print(f"\nTrain size: {len(X_train)}")
print(f"Valid size: {len(X_valid)}")
print(f"Test size : {len(X_test)}")

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nValid target distribution:")
print(y_valid.value_counts(normalize=True).sort_index())

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True).sort_index())

# Индексы категориальных признаков для CatBoost
cat_feature_indices = [X.columns.get_loc(col) for col in categorical_features]

# =========================
# 8. MODEL
# =========================
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    verbose=100,
)

# =========================
# 9. FIT
# =========================
model.fit(
    X_train,
    y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
    early_stopping_rounds=100,
)

# =========================
# 10. PREDICT ON TEST
# =========================
y_pred = model.predict(X_test)
y_pred = np.array(y_pred).astype(int).ravel()

y_proba = model.predict_proba(X_test)[:, 1]

# =========================
# 11. METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n=== CatBoost Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

# =========================
# 12. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title("Confusion Matrix - CatBoost")
plt.tight_layout()
plt.show()

# =========================
# 13. FEATURE IMPORTANCE
# =========================
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\n=== Top feature importances ===")
print(importance_df.head(20).to_string(index=False))

# =========================
# 14. PLOT TOP 15 IMPORTANCES
# =========================
top_n = 15
top_df = importance_df.head(top_n).sort_values("importance")

plt.figure(figsize=(10, 6))
plt.barh(top_df["feature"], top_df["importance"])
plt.title("Top 15 feature importances - CatBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
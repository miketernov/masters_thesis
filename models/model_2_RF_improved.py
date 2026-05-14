import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
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

# =========================
# 1. PATHS
# =========================
INPUT_FILE = r"xxx"

# =========================
# 2. SETTINGS
# =========================
TARGET_COL = "target"
TEST_SIZE = 0.2
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

# 1) restaurant rating relative to local competitors
df["rating_vs_comp"] = df["avg_rating"] - df["comp_rating_avg_300m"]

# 2) insufficient review volume flag
df["low_reviews"] = (df["reviews_count_collected"] < 20).astype(int)

# 3) overall complaint intensity
df["total_complaint_share"] = (
    df["service_complaint_share"].fillna(0)
    + df["food_complaint_share"].fillna(0)
    + df["price_complaint_share"].fillna(0)
)

# 4) log-transformed average bill
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
y = df[TARGET_COL].copy()

# =========================
# 6. IDENTIFY FEATURE TYPES
# =========================
numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [col for col in X.columns if col not in numeric_features]

print("Numeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# =========================
# 7. PREPROCESSING
# RF does not require scaling
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
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
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model),
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

print("\n=== Random Forest Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

# =========================
# 13. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()

# =========================
# 14. FEATURE IMPORTANCE
# =========================
feature_names = []
feature_names.extend(numeric_features)

if categorical_features:
    ohe = clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names.extend(cat_feature_names)

importances = clf.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\n=== Top feature importances ===")
print(importance_df.head(20).to_string(index=False))

# =========================
# 15. PLOT TOP 15 IMPORTANCES
# =========================
top_n = 15
top_df = importance_df.head(top_n).sort_values("importance")

plt.figure(figsize=(10, 6))
plt.barh(top_df["feature"], top_df["importance"])
plt.title("Top 15 feature importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
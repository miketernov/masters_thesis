import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
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

df["rating_vs_comp"] = df["avg_rating"] - df["comp_rating_avg_300m"]
df["low_reviews"] = (df["reviews_count_collected"] < 20).astype(int)
df["total_complaint_share"] = (
    df["service_complaint_share"].fillna(0)
    + df["food_complaint_share"].fillna(0)
    + df["price_complaint_share"].fillna(0)
)
df["log_avg_bill"] = np.log1p(df["avg_bill"])

# =========================
# 5. EXCLUDE NON-MODEL / REDUNDANT COLUMNS
# =========================
excluded_cols = {
    TARGET_COL,

    "restaurant_id",
    "date",
    "rest_name",
    "full_address",
    "latitude",
    "longitude",
    "metro_station",

    "rating_count",
    "review_count",

    "avg_bill",

    # sentiment
    "positive_share",
    "negative_share",
    # "neutral_share",
    # "avg_sentiment_score",

    # duplicate
    "avg_review_len_chars",

    # competition counts
    # "comp_100m",
    # "comp_200m",

    # competition ratings
    # "comp_rating_avg_100m",
    "comp_rating_avg_200m",

    # complaint-derived
    # "negative_service_share",
    # "negative_food_share",
    # "negative_price_share",

    # near-zero variance
    "flag_no_comp_200m",
}

feature_cols = [col for col in df.columns if col not in excluded_cols]

if not feature_cols:
    raise ValueError("No feature columns left after exclusions.")

X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int).copy()

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
# 7. TRAIN / TEST SPLIT
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
# 8. PREPROCESSING
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
# 9. MODEL + GRID SEARCH
# =========================
base_model = LogisticRegression(
    max_iter=5000,
    random_state=RANDOM_STATE,
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", base_model),
])

param_grid = [
    {
        "model__solver": ["liblinear"],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 0.5, 1, 2, 5, 10],
        "model__class_weight": [None, "balanced"],
    },
    {
        "model__solver": ["saga"],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 0.5, 1, 2, 5, 10],
        "model__class_weight": [None, "balanced"],
    },
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n=== Best params ===")
print(grid.best_params_)
print(f"Best CV F1: {grid.best_score_:.4f}")

# =========================
# 10. THRESHOLD SEARCH ON TRAIN (OOF)
# =========================
def search_best_threshold(y_true, y_proba, thresholds=np.arange(0.20, 0.81, 0.01)):
    rows = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    return pd.DataFrame(rows).sort_values("f1", ascending=False)

oof_proba = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

thr_df = search_best_threshold(y_train, oof_proba)

print("\n=== Top thresholds on TRAIN (OOF) by F1 ===")
print(thr_df.head(10).to_string(index=False))

best_threshold = thr_df.iloc[0]["threshold"]
print(f"\nBest threshold from TRAIN OOF: {best_threshold:.2f}")

# =========================
# 11. REFIT ON FULL TRAIN
# =========================
best_model.fit(X_train, y_train)

# =========================
# 12. FINAL EVALUATION ON TEST
# =========================
y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, zero_division=0)
rec = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

print("\n=== Logistic Regression Metrics (TEST) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n=== Classification Report (TEST) ===")
print(classification_report(y_test, y_test_pred, digits=4))

# =========================
# 13. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title(f"Confusion Matrix - Logistic Regression (thr={best_threshold:.2f})")
plt.tight_layout()
plt.show()

# =========================
# 14. FEATURE COEFFICIENTS
# =========================
feature_names = []
feature_names.extend(numeric_features)

if categorical_features:
    ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names.extend(cat_feature_names)

coefficients = best_model.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
})
coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
coef_df["abs_coef"] = coef_df["coefficient"].abs()

coef_df = coef_df.sort_values("coefficient", ascending=False)

print("\n=== Top positive coefficients (higher probability of target=1) ===")
print(coef_df.head(20)[["feature", "coefficient", "odds_ratio"]].to_string(index=False))

print("\n=== Top negative coefficients (lower probability of target=1) ===")
print(coef_df.tail(20)[["feature", "coefficient", "odds_ratio"]].to_string(index=False))

print("\n=== Strongest features by |coefficient| ===")
print(
    coef_df.sort_values("abs_coef", ascending=False)
    .head(20)[["feature", "coefficient", "odds_ratio"]]
    .to_string(index=False)
)
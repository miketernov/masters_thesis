"""
Модель предсказания раннего закрытия ресторанов в Москве.
XGBoost без SMOTE.
Честная схема: train / valid / test, threshold подбирается на valid.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
)
from xgboost import XGBClassifier

# =========================
# 1. LOAD DATA
# =========================
INPUT_FILE = "restaurants_with_review_features_v_5.csv"
TARGET_COL = "target"

df = pd.read_csv(INPUT_FILE)
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df[TARGET_COL].value_counts()}\n")

# =========================
# 2. FEATURE ENGINEERING
# =========================
df["rating_vs_comp"] = df["avg_rating"] - df["comp_rating_avg_300m"]
df["low_reviews"] = (df["reviews_count_collected"] < 20).astype(int)
df["total_complaint_share"] = (
    df["service_complaint_share"]
    + df["food_complaint_share"]
    + df["price_complaint_share"]
)
df["log_avg_bill"] = np.log1p(df["avg_bill"])

# =========================
# 3. FEATURE SELECTION
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
}

redundant_cols = {
    "positive_share",
    #"neutral_share",
    "negative_share",
    "avg_sentiment_score",
    "avg_review_len_chars",
    #"comp_100m",
    #"comp_200m",
    #"comp_rating_avg_100m",
    #"comp_rating_avg_200m",
    #"negative_service_share",
    #"negative_food_share",
    #"negative_price_share",
    "flag_no_comp_200m",
    "avg_bill",
}

feature_cols = [
    col for col in df.columns
    if col not in excluded_cols and col not in redundant_cols
]

X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int).copy()

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [col for col in X.columns if col not in numeric_features]

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}\n")

# =========================
# 4. TRAIN / VALID / TEST SPLIT
# =========================
TEST_SIZE = 0.2
VALID_SIZE = 0.2
RANDOM_STATE = 42

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

print(f"Train size: {len(X_train)}")
print(f"Valid size: {len(X_valid)}")
print(f"Test size : {len(X_test)}")

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nValid target distribution:")
print(y_valid.value_counts(normalize=True).sort_index())

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True).sort_index())

# =========================
# 5. PREPROCESSING
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# =========================
# 6. HELPER: THRESHOLD SEARCH
# =========================
def find_best_threshold(y_true, y_proba, lo=0.15, hi=0.65, step=0.01):
    best_f1 = 0
    best_thr = 0.5

    for t in np.arange(lo, hi, step):
        y_pred = (y_proba >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = t

    return best_thr, best_f1

def evaluate_with_threshold(name, y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)

    print(f"\n{'='*60}")
    print(f"{name} | threshold = {threshold:.2f}")
    print(f"{'='*60}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))

    return y_pred

# =========================
# 7. MODEL: XGBoost WITHOUT SMOTE
# =========================
pipe_xgb = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1,
    )),
])

grid_xgb = GridSearchCV(
    pipe_xgb,
    param_grid={
        "model__n_estimators": [150, 300],
        "model__max_depth": [4, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__min_child_weight": [5],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
    },
    scoring="f1",
    cv=cv,
    n_jobs=-1,
)

grid_xgb.fit(X_train, y_train)

print(f"\nXGBoost best params: {grid_xgb.best_params_}")
print(f"XGBoost best CV F1: {grid_xgb.best_score_:.4f}")

best_model = grid_xgb.best_estimator_

# =========================
# 8. THRESHOLD SELECTION ON VALID
# =========================
y_valid_proba = best_model.predict_proba(X_valid)[:, 1]
best_threshold, best_valid_f1 = find_best_threshold(y_valid, y_valid_proba)

print(f"\nBest threshold on VALID: {best_threshold:.2f}")
print(f"Best VALID F1 at this threshold: {best_valid_f1:.4f}")

# =========================
# 9. FINAL EVALUATION ON TEST
# =========================
y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = evaluate_with_threshold(
    "XGBoost (TEST)",
    y_test,
    y_test_proba,
    best_threshold,
)

# =========================
# 10. FEATURE IMPORTANCE
# =========================
feat_names = numeric_features.copy()

if categorical_features:
    ohe = (
        best_model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )
    feat_names += ohe.get_feature_names_out(categorical_features).tolist()

importances = best_model.named_steps["model"].feature_importances_

imp_df = pd.DataFrame({
    "feature": feat_names,
    "importance": importances,
}).sort_values("importance", ascending=False)

print("\n=== Top 15 Features (XGBoost) ===")
print(imp_df.head(15).to_string(index=False))

# =========================
# 11. PLOTS
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC curve on test
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
auc = roc_auc_score(y_test, y_test_proba)
axes[0].plot(fpr, tpr, label=f"XGBoost (AUC={auc:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_title("ROC Curve (Test)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Confusion matrix on test
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=axes[1])
axes[1].set_title("Confusion Matrix (Test)")

plt.tight_layout()
plt.savefig("xgboost_results_no_smote_valid_test.png", dpi=150)
print("\nPlots saved to xgboost_results_no_smote_valid_test.png")
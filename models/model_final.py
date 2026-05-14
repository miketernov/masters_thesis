import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =========================
# 1. LOAD DATA
# =========================
INPUT_FILE = r"xxx"
TARGET_COL = "target"

df = pd.read_csv(INPUT_FILE)

# =========================
# 2. FEATURE SELECTION
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
}

# Убираем явно избыточные признаки
redundant_cols = {
    # shares sum to 1 -> оставляем только negative_share
    "positive_share",
    "neutral_share",

    # конкуренция: оставляем 300м + флаги локального отсутствия
    "comp_100m",
    "comp_200m",
    "comp_rating_avg_100m",
    "comp_rating_avg_200m",

    # длина отзывов
    "avg_review_len_chars",

    # производные complaint/sentiment признаки
    "negative_service_share",
    "negative_food_share",
    "negative_price_share",
    "service_among_negative_share",
    "food_among_negative_share",
    "price_among_negative_share",
}

feature_cols = [
    col for col in df.columns
    if col not in excluded_cols and col not in redundant_cols
]

X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int).copy()

# =========================
# 3. LOG TRANSFORM SKEWED FEATURES
# =========================
log_cols = [
    "avg_bill",
    "rating_count",
    "review_count",
    "reviews_count_collected",
    "metro_distance",
    "center_distance",
    "comp_300m",
]

for col in log_cols:
    if col in X.columns:
        X[col] = np.log1p(X[col])

# =========================
# 4. FEATURE TYPES
# =========================
numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [col for col in X.columns if col not in numeric_features]

print("Numeric features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)

# =========================
# 5. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# =========================
# 6. PREPROCESSING
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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
# 7. MODEL PIPELINE
# =========================
pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=7000,
        solver="saga",
        random_state=42
    ))
])

# =========================
# 8. GRID SEARCH
# =========================
param_grid = [
    {
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        "model__class_weight": [None, "balanced"],
    },
    {
        "model__penalty": ["elasticnet"],
        "model__C": [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        "model__class_weight": [None, "balanced"],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="average_precision",   # лучше для дисбаланса
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

print("\nBest params:")
print(grid.best_params_)
print(f"Best CV PR-AUC: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# =========================
# 9. THRESHOLD SEARCH ONLY ON TRAIN (OOF)
# =========================
oof_proba = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

thresholds = np.arange(0.15, 0.81, 0.01)

thr_results = []
for thr in thresholds:
    y_pred_thr = (oof_proba >= thr).astype(int)
    thr_results.append({
        "threshold": thr,
        "precision": precision_score(y_train, y_pred_thr, zero_division=0),
        "recall": recall_score(y_train, y_pred_thr, zero_division=0),
        "f1": f1_score(y_train, y_pred_thr, zero_division=0),
    })

thr_df = pd.DataFrame(thr_results).sort_values("f1", ascending=False)
print("\nTop thresholds on TRAIN by F1:")
print(thr_df.head(10).to_string(index=False))

best_threshold = thr_df.iloc[0]["threshold"]
print(f"\nBest threshold from TRAIN CV by F1: {best_threshold:.2f}")

# =========================
# 10. FINAL FIT ON FULL TRAIN
# =========================
best_model.fit(X_train, y_train)

# =========================
# 11. TEST EVALUATION
# =========================
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= best_threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print("\n=== Tuned Logistic Regression Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

# =========================
# 12. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title(f"Confusion Matrix - Tuned Logistic Regression (thr={best_threshold:.2f})")
plt.tight_layout()
plt.show()

# =========================
# 13. COEFFICIENTS
# =========================
feature_names = []

# numeric names after imputer with indicators
num_transformer = best_model.named_steps["preprocessor"].named_transformers_["num"]
num_imputer = num_transformer.named_steps["imputer"]

num_feature_names = numeric_features.copy()
if hasattr(num_imputer, "indicator_") and num_imputer.indicator_ is not None:
    missing_idx = num_imputer.indicator_.features_
    for idx in missing_idx:
        num_feature_names.append(f"{numeric_features[idx]}_missing")

feature_names.extend(num_feature_names)

if categorical_features:
    ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names.extend(cat_feature_names)

coefficients = best_model.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", ascending=False)

print("\n=== Top positive coefficients ===")
print(coef_df.head(20).to_string(index=False))

print("\n=== Top negative coefficients ===")
print(coef_df.tail(20).to_string(index=False))
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
RANDOM_STATE = 42

# =========================
# 3. LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

if TARGET_COL not in df.columns:
    raise ValueError(f"Column '{TARGET_COL}' not found in dataset.")

# =========================
# 4. EXCLUDE NON-MODEL COLUMNS
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

feature_cols = [col for col in df.columns if col not in excluded_cols]

if not feature_cols:
    raise ValueError("No feature columns left after exclusions.")

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

# =========================
# 5. IDENTIFY CATEGORICAL FEATURES
# =========================
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Categorical features:")
print(categorical_features)

print("\nAll model features:")
print(X.columns.tolist())

# =========================
# 6. TRAIN / TEST SPLIT
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

# Индексы категориальных признаков для CatBoost
cat_feature_indices = [X.columns.get_loc(col) for col in categorical_features]

# =========================
# 7. MODEL
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
# 8. FIT
# =========================
model.fit(
    X_train,
    y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_test, y_test),
    use_best_model=True,
)

# =========================
# 9. PREDICT
# =========================
y_pred = model.predict(X_test)
y_pred = np.array(y_pred).astype(int).ravel()

y_proba = model.predict_proba(X_test)[:, 1]

# =========================
# 10. METRICS
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
# 11. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title("Confusion Matrix - CatBoost")
plt.tight_layout()
plt.show()

# =========================
# 12. FEATURE IMPORTANCE
# =========================
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\n=== Top feature importances ===")
print(importance_df.head(20).to_string(index=False))

# =========================
# 13. PLOT TOP 15 IMPORTANCES
# =========================
top_n = 15
top_df = importance_df.head(top_n).sort_values("importance")

plt.figure(figsize=(10, 6))
plt.barh(top_df["feature"], top_df["importance"])
plt.title("Top 15 feature importances - CatBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
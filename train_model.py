
# SMART DIET RECOMMENDATION SYSTEM
# Model Training, Evaluation and Saving

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from features import RAW_FEATURES, TARGET, add_engineered_features

warnings.filterwarnings("ignore")

# 1. CONFIGURATION

DATASET_PATH = r"C:\Users\h\diet_recommendation_dataset_1000.csv"
MODEL_PATH = "diet_model.joblib"
METRICS_PATH = "model_metrics.txt"
EDA_FOLDER = "eda_outputs"

RANDOM_STATE = 42

os.makedirs(EDA_FOLDER, exist_ok=True)


# 2. LOAD DATASET
print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place diet_recommendation_dataset_1000.csv "
        "in the same folder as train_model.py."
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


# 3. CHECK REQUIRED COLUMNS

required_columns = RAW_FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )

df = df[required_columns].copy()


# 4. DATA CLEANING

print("\nCleaning dataset...")

# Replace infinite numeric values with missing values
df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows without a target label
df = df.dropna(subset=[TARGET])

# Remove duplicate records
df = df.drop_duplicates()

# Standardize target labels
df[TARGET] = df[TARGET].astype(str).str.strip()

# Remove empty target labels
df = df[df[TARGET] != ""]

# Convert expected numeric columns to numeric format
numeric_input_columns = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Sugar_Level",
    "Cholesterol",
]

for col in numeric_input_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing values in essential numeric inputs
df = df.dropna(subset=numeric_input_columns)

if df.empty:
    raise ValueError("No valid rows remain after data cleaning.")

print("Cleaned dataset shape:", df.shape)
print("\nDiet class distribution:")
print(df[TARGET].value_counts())


# 5. EXPLORATORY DATA ANALYSIS (EDA)

print("\nGenerating EDA charts...")

# Chart 1: Distribution of diet categories
plt.figure(figsize=(8, 5))

df[TARGET].value_counts().plot(kind="bar")

plt.title("Distribution of Diet Categories")
plt.xlabel("Diet Category")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "diet_distribution.png")
)

plt.close()


# Chart 2: BMI distribution
plt.figure(figsize=(8, 5))

df["BMI"].plot(kind="hist", bins=20)

plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "bmi_distribution.png")
)

plt.close()


# Chart 3: Correlation heatmap for numeric columns
numeric_df = df[numeric_input_columns].copy()

correlation = numeric_df.corr()

plt.figure(figsize=(9, 6))

plt.imshow(correlation, aspect="auto")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right",
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns,
)

plt.title("Correlation Between Numeric Features")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "correlation_heatmap.png")
)

plt.close()

print("EDA charts saved in:", EDA_FOLDER)


# ============================================================
# 6. DEFINE FEATURES (X) AND TARGET (y)
# ============================================================

X = df[RAW_FEATURES].copy()
y = df[TARGET].copy()

print("\nInput features (X):")
print(X.head())

print("\nTarget variable (y):")
print(y.head())


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

# Ensure each class has enough records for a stratified split
class_counts = y.value_counts()

if len(class_counts) < 2:
    raise ValueError(
        "At least two diet categories are required for classification."
    )

if class_counts.min() < 2:
    raise ValueError(
        "Each diet category needs at least two records "
        "for a stratified train-test split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 8. FEATURE ENGINEERING
X_train = add_engineered_features(X_train)
X_test = add_engineered_features(X_test)

print("\nEngineered training features:")
print(X_train.head())


# 9. DEFINE NUMERIC AND CATEGORICAL FEATURES

numeric_features = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Sugar_Level",
    "Cholesterol",
    "Weight_Height_Ratio",
    "Age_BMI_Interaction",
]

categorical_features = [
    "Gender",
    "Activity_Level",
    "Goal",
    "Age_Group",
    "BMI_Category",
    "Activity_Goal_Combination",
]


# 10. DATA PREPROCESSING

numeric_transformer = SklearnPipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = SklearnPipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features,
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# 11. DEFINE MACHINE LEARNING MODELS

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    ),
}


# 12. STRATIFIED CROSS-VALIDATION

# Use only folds for which every training class has enough
# examples to apply SMOTE with k_neighbors=1.

minimum_class_count = y_train.value_counts().min()

n_splits = min(5, int(minimum_class_count))

if n_splits < 2:
    raise ValueError(
        "Insufficient training examples per class for "
        "cross-validation and SMOTE."
    )

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print(f"\nUsing {n_splits}-fold stratified cross-validation.")


# 13. COMPARE MODELS USING CROSS-VALIDATION

cv_results = {}

print("\nComparing machine learning models...")

for model_name, classifier in models.items():

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "feature_selection",
                SelectPercentile(
                    score_func=partial(
                        mutual_info_classif,
                        random_state=RANDOM_STATE,
                    ),
                    percentile=80,
                ),
            ),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=1,
                ),
            ),
            ("classifier", classifier),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    cv_results[model_name] = scores.mean()

    print(
        f"{model_name}: "
        f"Mean weighted F1 = {scores.mean():.4f}"
    )


# 14. SELECT BEST MODEL

best_model_name = max(
    cv_results,
    key=cv_results.get,
)

print("\nBest model from cross-validation:", best_model_name)


# 15. HYPERPARAMETER TUNING

print("\nTuning the selected model...")

best_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "feature_selection",
            SelectPercentile(
                score_func=partial(
                    mutual_info_classif,
                    random_state=RANDOM_STATE,
                ),
                percentile=80,
            ),
        ),
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=1,
            ),
        ),
        (
            "classifier",
            models[best_model_name],
        ),
    ]
)

if best_model_name == "Logistic Regression":

    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
    }

elif best_model_name == "Random Forest":

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
    }

else:

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
    }

grid_search = GridSearchCV(
    estimator=best_pipeline,
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

grid_search.fit(X_train, y_train)

final_model = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
print("Best CV weighted F1:", round(grid_search.best_score_, 4))


# 16. PREDICT ON THE TEST DATA

y_pred = final_model.predict(X_test)


# 17. MODEL EVALUATION

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

print("\n========== TEST RESULTS ==========")

print(f"Selected model: {best_model_name}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    zero_division=0,
)

print(report)



# 18. CONFUSION MATRIX

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation="vertical",
)


plt.title("Diet Recommendation Model - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(EDA_FOLDER, "confusion_matrix.png")
)

plt.close()


# 19. SAVE TRAINED MODEL

joblib.dump(final_model, MODEL_PATH)

print(f"\nTrained model saved to: {MODEL_PATH}")


# 20. SAVE MODEL METRICS

with open(METRICS_PATH, "w", encoding="utf-8") as file:

    file.write("SMART DIET RECOMMENDATION SYSTEM\n")
    file.write("================================\n\n")

    file.write(f"Selected model: {best_model_name}\n")
    file.write(f"Best parameters: {grid_search.best_params_}\n")
    file.write(f"Best CV weighted F1: {grid_search.best_score_:.4f}\n\n")

    file.write(f"Test accuracy: {accuracy:.4f}\n")
    file.write(f"Test precision: {precision:.4f}\n")
    file.write(f"Test recall: {recall:.4f}\n")
    file.write(f"Test F1-score: {f1:.4f}\n\n")

    file.write("Cross-validation results:\n")

    for name, score in cv_results.items():
        file.write(f"{name}: {score:.4f}\n")

    file.write("\nClassification Report:\n")
    file.write(report)

print(f"Evaluation metrics saved to: {METRICS_PATH}")

print("\nModel training and evaluation completed successfully!")
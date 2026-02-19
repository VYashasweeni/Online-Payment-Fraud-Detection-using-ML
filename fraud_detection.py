# ============================================
# Online Payments Fraud Detection using ML
# ============================================

# 1. Import Required Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

import pickle


# 2. Check Working Directory & Files
print("Current Directory:", os.getcwd())
print("Files in Directory:", os.listdir())


# 3. Load Dataset
data = pd.read_csv("DatasetOP.csv")
print("\nDataset Loaded Successfully")
print("Shape:", data.shape)
print(data.head())


# 4. Target Column
TARGET_COLUMN = "isFraud"
print("\nTarget Column:", TARGET_COLUMN)


# 5. Check Missing Values
print("\nMissing Values Check:")
print(data.isnull().sum())


# 6. Encode Categorical (Object) Columns
label_encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = label_encoder.fit_transform(data[col])

print("\nCategorical columns encoded successfully")


# 7. Outlier Handling using IQR (SAFE for Python 3.7)

exclude_cols = ["isFraud", "type", "nameOrig", "nameDest"]

numerical_cols = [col for col in data.columns if col not in exclude_cols]

for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[col] = np.where(
        data[col] < lower_bound, lower_bound,
        np.where(data[col] > upper_bound, upper_bound, data[col])
    )

print("\nOutlier handling completed using IQR method")



# 8. Feature and Target Split
feature_columns = [
    'step',
    'type',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'isFlaggedFraud'
]

X = data[feature_columns]
y = data[TARGET_COLUMN]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# 9. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Done")
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# 10. Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}


# 11. Train and Evaluate Models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# 12. Select Best Model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", results[best_model_name])


# 13. Confusion Matrix
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 14. Save Model
with open("fraud_detection_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nModel saved as fraud_detection_model.pkl")

# ============================================
# END OF PROJECT
# ============================================

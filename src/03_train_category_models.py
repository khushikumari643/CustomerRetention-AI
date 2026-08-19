"""
CustomerRetention AI - Step 3: Customer Categorization Model Training
------------------------------------------------------------------------
Multiclass classification predicting Customer_Category
(High-Risk / Medium-Risk / Loyal / Premium-Ready) using 3 algorithms:
  1. Logistic Regression (multinomial baseline)
  2. Decision Tree (rule-like, interpretable)
  3. Random Forest (ensemble, usually strongest)

Selects the best model by accuracy on a held-out test set.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "/home/claude/CustomerRetentionAI/data/processed/customer_data_clean.csv"
MODEL_DIR = "/home/claude/CustomerRetentionAI/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Note: we deliberately EXCLUDE Customer_Status / Churn_Label / Tenure-derived
# leakage of the exact rule so the models learn generalizable patterns from
# raw behavioral + billing features (not just re-deriving the business rule).
FEATURES = [
    "Age", "Married", "Number_of_Referrals", "Tenure_in_Months", "Gender",
    "Phone_Service", "Multiple_Lines", "Internet_Service", "Internet_Type",
    "Online_Security", "Online_Backup", "Device_Protection_Plan",
    "Premium_Support", "Streaming_TV", "Streaming_Movies", "Streaming_Music",
    "Unlimited_Data", "Contract", "Paperless_Billing", "Payment_Method",
    "Monthly_Charge", "Total_Charges", "Total_Revenue",
    "Total_Addon_Services", "Avg_Monthly_Revenue", "Has_Value_Deal",
    "Is_High_Spender", "Has_Referrals", "Is_Senior", "Value_Deal", "State"
]
TARGET = "Customer_Category"

CATEGORICAL = [
    "Married", "Gender", "Phone_Service", "Multiple_Lines", "Internet_Service",
    "Internet_Type", "Online_Security", "Online_Backup", "Device_Protection_Plan",
    "Premium_Support", "Streaming_TV", "Streaming_Movies", "Streaming_Music",
    "Unlimited_Data", "Contract", "Paperless_Billing", "Payment_Method",
    "Value_Deal", "State"
]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def main():
    df = pd.read_csv(DATA_PATH)

    encoders = {}
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    y = le_target.fit_transform(df[TARGET])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = X_tr.copy()
    X_te_s = X_te.copy()
    X_tr_s[NUMERIC] = scaler.fit_transform(X_tr[NUMERIC])
    X_te_s[NUMERIC] = scaler.transform(X_te[NUMERIC])

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)
        acc = accuracy_score(y_te, preds)
        results[name] = {"accuracy": round(acc, 4)}
        fitted[name] = model
        print(f"{name}: accuracy={acc:.4f}")
        print(classification_report(y_te, preds, target_names=le_target.classes_))

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = fitted[best_name]
    print(f"\nBest categorization model: {best_name} -> {results[best_name]}")

    joblib.dump(best_model, f"{MODEL_DIR}/best_category_model_{best_name}.pkl")
    joblib.dump(best_model, f"{MODEL_DIR}/best_category_model.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/category_encoders.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/category_scaler.pkl")
    joblib.dump(le_target, f"{MODEL_DIR}/category_label_encoder.pkl")

    with open(f"{MODEL_DIR}/category_model_results.json", "w") as f:
        json.dump({"results": results, "best_model": best_name, "features": FEATURES,
                    "categorical": CATEGORICAL, "numeric": NUMERIC,
                    "classes": le_target.classes_.tolist()}, f, indent=2)

    print("\nAll categorization artifacts saved to", MODEL_DIR)


if __name__ == "__main__":
    main()

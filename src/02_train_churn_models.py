"""
CustomerRetention AI - Step 2: Churn Prediction Model Training
------------------------------------------------------------------
Trains 3 ML algorithms for binary churn classification:
  1. Logistic Regression (baseline, interpretable)
  2. Random Forest (non-linear patterns)
  3. XGBoost (strong predictive performance)

Also trains a secondary multiclass model on the CHURNED subset only to
predict Churn_Category (the diagnostic "reason" driver) so the app can
explain WHY a customer is likely to churn.

Saves the best churn model + a feature-importance based explainer info.
"""



#MAKE sure to change the path address based on local directory
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

DATA_PATH=r"C:\Users\Khush\OneDrive\Desktop\web\CustomerRetentionAI\data\processed\customer_data_clean.csv"
MODEL_DIR=r"C:\Users\Khush\OneDrive\Desktop\web\CustomerRetentionAI\models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES=[
    "Age", "Married", "Number_of_Referrals", "Tenure_in_Months", "Gender",
    "Phone_Service", "Multiple_Lines", "Internet_Service", "Internet_Type",
    "Online_Security", "Online_Backup", "Device_Protection_Plan",
    "Premium_Support", "Streaming_TV", "Streaming_Movies", "Streaming_Music",
    "Unlimited_Data", "Contract", "Paperless_Billing", "Payment_Method",
    "Monthly_Charge", "Total_Charges", "Total_Refunds",
    "Total_Extra_Data_Charges", "Total_Long_Distance_Charges",
    "Total_Revenue", "Total_Addon_Services", "Avg_Monthly_Revenue",
    "Extra_Charge_Ratio", "Refund_Ratio", "Is_Long_Term_Contract",
    "Has_Value_Deal", "Is_High_Spender", "Has_Referrals", "Is_Senior",
    "Value_Deal", "State"
]
TARGET="Churn_Label"

CATEGORICAL=[
    "Married", "Gender", "Phone_Service", "Multiple_Lines", "Internet_Service",
    "Internet_Type", "Online_Security", "Online_Backup", "Device_Protection_Plan",
    "Premium_Support", "Streaming_TV", "Streaming_Movies", "Streaming_Music",
    "Unlimited_Data", "Contract", "Paperless_Billing", "Payment_Method",
    "Value_Deal", "State"
]
NUMERIC=[c for c in FEATURES if c not in CATEGORICAL]


def load_and_split():
    df=pd.read_csv(DATA_PATH)
    train_df=df[df["Customer_Status"]!="Joined"].copy()
    return df,train_df


def build_encoders(train_df):
    encoders={}
    for col in CATEGORICAL:
        le=LabelEncoder()
        le.fit(train_df[col].astype(str))
        encoders[col]=le
    return encoders


def transform(df_subset, encoders, scaler=None, fit_scaler=False):
    X=df_subset[FEATURES].copy()
    for col in CATEGORICAL:
        le=encoders[col]
        X[col]=X[col].astype(str).map(
            lambda v:v if v in le.classes_ else le.classes_[0]
        )
        X[col]=le.transform(X[col])
    if fit_scaler:
        scaler=StandardScaler()
        X[NUMERIC]=scaler.fit_transform(X[NUMERIC])
    else:
        X[NUMERIC]=scaler.transform(X[NUMERIC])
    return X, scaler


def main():
    df,train_df=load_and_split()
    encoders=build_encoders(train_df)

    X_train_full,y=train_df[FEATURES], train_df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        train_df, train_df[TARGET], test_size=0.2, random_state=42, stratify=train_df[TARGET]
    )

    X_tr_enc,scaler=transform(X_tr, encoders, fit_scaler=True)
    X_te_enc, _=transform(X_te, encoders, scaler=scaler, fit_scaler=False)

    models={
        "Logistic_Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum()
        ),
    }

    results={}
    fitted={}
    for name, model in models.items():
        model.fit(X_tr_enc, y_tr)
        preds=model.predict(X_te_enc)
        probs=model.predict_proba(X_te_enc)[:, 1]
        results[name]={
            "accuracy": round(accuracy_score(y_te, preds), 4),
            "precision": round(precision_score(y_te, preds), 4),
            "recall": round(recall_score(y_te, preds), 4),
            "f1_score": round(f1_score(y_te, preds), 4),
            "roc_auc": round(roc_auc_score(y_te, probs), 4),
        }
        fitted[name]=model
        print(f"{name}:{results[name]}")

    best_name=max(results,key=lambda n:results[n]["roc_auc"])
    best_model=fitted[best_name]
    print(f"\nBest churn model:{best_name} = {results[best_name]}")

    # Save best model + artifacts
    joblib.dump(best_model, f"{MODEL_DIR}/best_churn_model_{best_name}.pkl")
    joblib.dump(best_model, f"{MODEL_DIR}/best_churn_model.pkl")  # canonical name
    joblib.dump(encoders, f"{MODEL_DIR}/churn_encoders.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/churn_scaler.pkl")

    with open(f"{MODEL_DIR}/churn_model_results.json", "w") as f:
        json.dump({"results": results, "best_model": best_name, "features": FEATURES,
                    "categorical": CATEGORICAL, "numeric": NUMERIC}, f, indent=2)

    # Feature importance for diagnostic explanations
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(FEATURES, best_model.feature_importances_.tolist()))
    else:
        importances = dict(zip(FEATURES, np.abs(best_model.coef_[0]).tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
    with open(f"{MODEL_DIR}/churn_feature_importance.json", "w") as f:
        json.dump(importances, f, indent=2)

    churned_df=df[df["Customer_Status"] == "Churned"].copy()
    cat_encoders=build_encoders(churned_df)
    le_reason=LabelEncoder()
    y_reason=le_reason.fit_transform(churned_df["Churn_Category"].astype(str))

    Xr_tr,Xr_te,yr_tr,yr_te=train_test_split(
        churned_df, y_reason, test_size=0.2, random_state=42, stratify=y_reason
    )
    Xr_tr_enc,reason_scaler=transform(Xr_tr, cat_encoders, fit_scaler=True)
    Xr_te_enc,_= transform(Xr_te, cat_encoders, scaler=reason_scaler, fit_scaler=False)

    reason_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1)
    reason_model.fit(Xr_tr_enc,yr_tr)
    reason_preds = reason_model.predict(Xr_te_enc)
    reason_acc = accuracy_score(yr_te, reason_preds)
    print(f"\nChurn-Reason (diagnostic) model accuracy: {reason_acc:.4f}")

    joblib.dump(reason_model, f"{MODEL_DIR}/churn_reason_model.pkl")
    joblib.dump(cat_encoders, f"{MODEL_DIR}/churn_reason_encoders.pkl")
    joblib.dump(reason_scaler, f"{MODEL_DIR}/churn_reason_scaler.pkl")
    joblib.dump(le_reason, f"{MODEL_DIR}/churn_reason_label_encoder.pkl")

    with open(f"{MODEL_DIR}/churn_reason_results.json", "w") as f:
        json.dump({"accuracy": round(reason_acc, 4),
                    "classes": le_reason.classes_.tolist()}, f, indent=2)

    print("\nAll churn artifacts saved to", MODEL_DIR)


if __name__ == "__main__":
    main()

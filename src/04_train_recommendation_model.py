"""
CustomerRetention AI - Step 4: Recommendation Engine Training
--------------------------------------------------------------------
Uses ONE ML algorithm (Random Forest) trained on the custom
subscription_plans.csv to recommend the best plan given a customer's
complaint/need profile.

Two outputs are modeled together:
  - Recommended_Plan   (classification target)
  - Recommended_Price  (regression target -> evaluated with MSE / R2)

Additionally runs a Chi-Squared test of independence between the
customer's Primary_Complaint and the Recommended_Plan to statistically
confirm the two are significantly associated (i.e. the complaint
meaningfully drives the recommendation, not random noise).
"""


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from scipy.stats import chi2_contingency

DATA_PATH=r"C:\Users\Khush\OneDrive\Desktop\web\CustomerRetentionAI\data\subscription_plans.csv"
MODEL_DIR=r"C:\Users\Khush\OneDrive\Desktop\web\CustomerRetentionAI\models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES=["Primary_Complaint", "Contract_Preference", "Monthly_Charge_Band",
            "Internet_Type_Need", "Family_Size_Need"]


def main():
    df=pd.read_csv(DATA_PATH)

    encoders={}
    X=df[FEATURES].copy()
    for col in FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    le_plan=LabelEncoder()
    y_plan=le_plan.fit_transform(df["Recommended_Plan"])
    y_price=df["Recommended_Price"].values

    # Small custom dataset (52 rows) -> 5-fold cross-validation gives a much
    # more reliable estimate than a single fragile holdout split.
    kf=KFold(n_splits=5, shuffle=True, random_state=42)

    clf=RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    plan_cv_preds=cross_val_predict(clf, X, y_plan, cv=kf)
    plan_acc=accuracy_score(y_plan, plan_cv_preds)

    reg=RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    price_cv_preds=cross_val_predict(reg, X, y_price, cv=kf)
    mse=mean_squared_error(y_price, price_cv_preds)
    rmse=np.sqrt(mse)
    r2=r2_score(y_price, price_cv_preds)

    # Fit final models on ALL data for deployment
    clf.fit(X, y_plan)
    reg.fit(X, y_price)

 
    contingency=pd.crosstab(df["Primary_Complaint"], df["Recommended_Plan"])
    chi2, p_value, dof, expected=chi2_contingency(contingency)

    print(f"Recommendation classifier accuracy: {plan_acc:.4f}")
    print(f"Price regressor -> MSE: {mse:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
    print(f"Chi-Squared (Complaint vs Plan): chi2={chi2:.3f}, p-value={p_value:.4f}, dof={dof}")

    joblib.dump(clf, f"{MODEL_DIR}/recommendation_model.pkl")
    joblib.dump(reg, f"{MODEL_DIR}/recommendation_price_model.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/recommendation_encoders.pkl")
    joblib.dump(le_plan, f"{MODEL_DIR}/recommendation_plan_label_encoder.pkl")

    with open(f"{MODEL_DIR}/recommendation_model_results.json", "w") as f:
        json.dump({
            "algorithm": "Random Forest (classifier + regressor)",
            "classification_accuracy": round(float(plan_acc), 4),
            "regression_mse": round(float(mse), 4),
            "regression_rmse": round(float(rmse), 4),
            "regression_r2": round(float(r2), 4),
            "chi_squared": round(float(chi2), 4),
            "chi_squared_p_value": round(float(p_value), 6),
            "chi_squared_dof": int(dof),
            "features": FEATURES,
        }, f, indent=2)

    print("\nRecommendation engine artifacts saved to", MODEL_DIR)


if __name__=="__main__":
    main()

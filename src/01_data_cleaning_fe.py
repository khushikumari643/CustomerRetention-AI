"""
CustomerRetention AI - Step 1: Data Cleaning & Feature Engineering
--------------------------------------------------------------------
Business Understanding (recap):
  Goal: Predict customer churn, diagnose WHY a customer is likely to churn,
  categorize customers by retention/purchase potential, and recommend the
  best subscription plan to retain them. 
  1. Load the raw dataset
  2. Handle missing values (business-logic driven, not blind imputation)
  3. Clean invalid values (e.g. negative charges)
  4. Engineer new features (tenure buckets, service counts, avg charge, etc.)-total 13 new categorical features added love......
  5. Create the two supervised business labels needed later:
       - Churn_Label (binary: Stayed/Joined = 0, Churned = 1)
       - Customer_Category (business rule -> multiclass label used to TRAIN
         the categorization models, exactly like Churn_Category is a label
         mined from business logic)
  6. Save the cleaned + engineered dataset to data/processed/
"""

import pandas as pd
import numpy as np

RAW_PATH = "raw data path"
OUT_PATH = "clean data path where you would like to save"

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape}")
    return df

def clean_data(df):
    df = df.copy()


    df.loc[df["Monthly_Charge"] < 0, "Monthly_Charge"] = np.nan
    df["Monthly_Charge"] = df["Monthly_Charge"].fillna(df["Monthly_Charge"].median())

  
    df["Value_Deal"] = df["Value_Deal"].fillna("No Deal")


    df["Multiple_Lines"] = df["Multiple_Lines"].fillna("No")

    internet_addons = ["Internet_Type", "Online_Security", "Online_Backup",
                        "Device_Protection_Plan", "Premium_Support", "Streaming_TV",
                        "Streaming_Movies", "Streaming_Music", "Unlimited_Data"]
    for col in internet_addons:
        if col == "Internet_Type":
            df[col] = df[col].fillna("No Internet")
        else:
            df[col] = df[col].fillna("No")

    
    for col in ["Total_Charges", "Total_Refunds", "Total_Extra_Data_Charges",
                "Total_Long_Distance_Charges", "Total_Revenue"]:
        df[col] = df[col].clip(lower=0)

    before = len(df)
    df = df.drop_duplicates(subset=["Customer_ID"])
    print(f"Dropped {before - len(df)} duplicate rows")

    return df

def engineer_features(df):
    df = df.copy()


    df["Churn_Label"] = (df["Customer_Status"] == "Churned").astype(int)

    df["Tenure_Bucket"] = pd.cut(
        df["Tenure_in_Months"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49mo+"]
    )

    addon_cols = ["Online_Security", "Online_Backup", "Device_Protection_Plan",
                  "Premium_Support", "Streaming_TV", "Streaming_Movies",
                  "Streaming_Music", "Unlimited_Data"]
    df["Total_Addon_Services"] = (df[addon_cols] == "Yes").sum(axis=1)

    df["Avg_Monthly_Revenue"] = df["Total_Revenue"] / df["Tenure_in_Months"].replace(0, 1)


    df["Extra_Charges_Total"] = df["Total_Extra_Data_Charges"] + df["Total_Long_Distance_Charges"]
    df["Extra_Charge_Ratio"] = df["Extra_Charges_Total"] / df["Total_Revenue"].replace(0, 1)

    df["Refund_Ratio"] = df["Total_Refunds"] / df["Total_Revenue"].replace(0, 1)

    df["Is_Long_Term_Contract"] = (df["Contract"] != "Month-to-Month").astype(int
    df["Has_Value_Deal"] = (df["Value_Deal"] != "No Deal").astype(int)

    q75 = df["Monthly_Charge"].quantile(0.75)
    df["Is_High_Spender"] = (df["Monthly_Charge"] >= q75).astype(int)


    df["Has_Referrals"] = (df["Number_of_Referrals"] > 0).astype(int)

    df["Is_Senior"] = (df["Age"] >= 60).astype(int)


    def categorize(row):
        if row["Customer_Status"] == "Churned":
            return "High-Risk"
        # Stayed / Joined customers
        revenue_q = df["Total_Revenue"].quantile(0.66)
        if row["Total_Revenue"] >= revenue_q and row["Is_Long_Term_Contract"] == 1:
            return "Premium-Ready"
        if row["Is_Long_Term_Contract"] == 1 and row["Tenure_in_Months"] >= 24:
            return "Loyal"
        if row["Contract"] == "Month-to-Month" and row["Tenure_in_Months"] < 12:
            return "Medium-Risk"
        return "Loyal" if row["Total_Addon_Services"] >= 3 else "Medium-Risk"

    df["Customer_Category"] = df.apply(categorize, axis=1)

    return df

def main():
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    import os
    os.makedirs("paste the  directory to save processed data ", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned + engineered data: {df.shape} -> {OUT_PATH}")

    print("\nChurn_Label distribution:\n", df["Churn_Label"].value_counts())
    print("\nCustomer_Category distribution:\n", df["Customer_Category"].value_counts())

if __name__ == "__main__":
    main()

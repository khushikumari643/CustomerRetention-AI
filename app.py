"""
CustomerRetention AI
=====================================================================
A Streamlit web application for telecom customer churn prediction,
diagnostic root-cause analysis, customer categorization, subscription
recommendation, and interactive business intelligence dashboarding.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CustomerRetention AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "processed" / "customer_data_clean.csv"
PLANS_PATH = BASE / "data" / "subscription_plans.csv"
MODEL_DIR = BASE / "models"

# ---------------------------------------------------------------------------
# Caching: data + models load once
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def load_plans():
    return pd.read_csv(PLANS_PATH)

@st.cache_resource
def load_churn_artifacts():
    model = joblib.load(MODEL_DIR / "best_churn_model.pkl")
    encoders = joblib.load(MODEL_DIR / "churn_encoders.pkl")
    scaler = joblib.load(MODEL_DIR / "churn_scaler.pkl")
    with open(MODEL_DIR / "churn_model_results.json") as f:
        results = json.load(f)
    with open(MODEL_DIR / "churn_feature_importance.json") as f:
        importance = json.load(f)
    return model, encoders, scaler, results, importance

@st.cache_resource
def load_churn_reason_artifacts():
    model = joblib.load(MODEL_DIR / "churn_reason_model.pkl")
    encoders = joblib.load(MODEL_DIR / "churn_reason_encoders.pkl")
    scaler = joblib.load(MODEL_DIR / "churn_reason_scaler.pkl")
    label_enc = joblib.load(MODEL_DIR / "churn_reason_label_encoder.pkl")
    with open(MODEL_DIR / "churn_reason_results.json") as f:
        results = json.load(f)
    return model, encoders, scaler, label_enc, results

@st.cache_resource
def load_category_artifacts():
    model = joblib.load(MODEL_DIR / "best_category_model.pkl")
    encoders = joblib.load(MODEL_DIR / "category_encoders.pkl")
    scaler = joblib.load(MODEL_DIR / "category_scaler.pkl")
    label_enc = joblib.load(MODEL_DIR / "category_label_encoder.pkl")
    with open(MODEL_DIR / "category_model_results.json") as f:
        results = json.load(f)
    return model, encoders, scaler, label_enc, results

@st.cache_resource
def load_recommendation_artifacts():
    clf = joblib.load(MODEL_DIR / "recommendation_model.pkl")
    reg = joblib.load(MODEL_DIR / "recommendation_price_model.pkl")
    encoders = joblib.load(MODEL_DIR / "recommendation_encoders.pkl")
    plan_le = joblib.load(MODEL_DIR / "recommendation_plan_label_encoder.pkl")
    with open(MODEL_DIR / "recommendation_model_results.json") as f:
        results = json.load(f)
    return clf, reg, encoders, plan_le, results

df = load_data()
plans_df = load_plans()

churn_model, churn_encoders, churn_scaler, churn_results, churn_importance = load_churn_artifacts()
reason_model, reason_encoders, reason_scaler, reason_le, reason_results = load_churn_reason_artifacts()
cat_model, cat_encoders, cat_scaler, cat_le, cat_results = load_category_artifacts()
rec_clf, rec_reg, rec_encoders, rec_plan_le, rec_results = load_recommendation_artifacts()

with open(MODEL_DIR / "churn_model_results.json") as f:
    CHURN_META = json.load(f)
with open(MODEL_DIR / "category_model_results.json") as f:
    CAT_META = json.load(f)

CHURN_FEATURES = CHURN_META["features"]
CHURN_CATEGORICAL = CHURN_META["categorical"]
CHURN_NUMERIC = CHURN_META["numeric"]

CAT_FEATURES = CAT_META["features"]
CAT_CATEGORICAL = CAT_META["categorical"]
CAT_NUMERIC = CAT_META["numeric"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def encode_row(row_dict, feature_list, categorical_list, encoders, scaler, numeric_list):
    X = pd.DataFrame([row_dict])[feature_list].copy()
    for col in categorical_list:
        le = encoders[col]
        val = str(X.at[0, col])
        if val not in le.classes_:
            val = le.classes_[0]
        X[col] = le.transform([val])
    X[numeric_list] = scaler.transform(X[numeric_list])
    return X

def get_customer_row(customer_id):
    row = df[df["Customer_ID"] == customer_id]
    if row.empty:
        return None
    return row.iloc[0]

CHURN_REASON_EXPLANATIONS = {
    "Competitor": "This customer profile resembles others who left for a competitor offering better devices, pricing, or data plans.",
    "Attitude": "Service/support interaction quality appears to be the main risk factor — attitude of support or service staff.",
    "Dissatisfaction": "General service or product dissatisfaction (reliability, network quality) is the dominant risk factor.",
    "Price": "Price sensitivity is the leading risk factor — monthly charges may be too high relative to perceived value.",
    "Other": "Risk factors are mixed / not dominated by a single category — recommend a general retention outreach.",
}

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fb;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #eaeaea;
    }
    .big-font { font-size:22px !important; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📡 CustomerRetention AI")
st.caption(
    "Churn Prediction • Diagnostic Root-Cause Analysis • Customer Categorization • "
    "Subscription Recommendation • Interactive Dashboard"
)

with st.sidebar:
    st.header("Navigation")
    section = st.radio(
        "Choose a section",
        [
            "1️⃣ Churn Prediction & Diagnosis",
            "2️⃣ Customer Categorization",
            "3️⃣ Recommendation Engine",
            "4️⃣ Interactive Dashboard",
        ],
    )
    st.markdown("---")
    st.markdown("### About this project")
    st.markdown(
        "CustomerRetention AI is a mini end-to-end ML project covering business "
        "understanding, data cleaning, feature engineering, EDA, multi-model "
        "training/selection, and deployment via Streamlit."
    )
    st.markdown(f"**Dataset size:** {len(df):,} customers")
    st.markdown(f"**Churn model:** {CHURN_META['best_model']} (ROC-AUC {churn_results['results'][CHURN_META['best_model']]['roc_auc']})")
    st.markdown(f"**Category model:** {CAT_META['best_model']} (Acc {cat_results['results'][CAT_META['best_model']]['accuracy']})")

# =============================================================================
# SECTION 1: CHURN PREDICTION & DIAGNOSTIC ANALYSIS
# =============================================================================
if section.startswith("1"):
    st.header("1️⃣ Churn Prediction & Diagnostic (Root-Cause) Analysis")
    st.write(
        "Select an existing customer or enter a new profile to predict churn "
        "probability and see the top factors driving that risk."
    )

    mode = st.radio("Input mode", ["Pick existing customer", "Enter new customer profile"], horizontal=True)

    if mode == "Pick existing customer":
        cust_id = st.selectbox("Select Customer ID", df["Customer_ID"].sample(300, random_state=1).sort_values())
        row = get_customer_row(cust_id)
        input_data = row.to_dict()
    else:
        st.subheader("Enter customer profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 100, 35)
            married = st.selectbox("Married", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            referrals = st.number_input("Number of Referrals", 0, 20, 0)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
            internet = st.selectbox("Internet Service", ["Yes", "No"])
            internet_type = st.selectbox("Internet Type", ["Fiber Optic", "Cable", "DSL", "No Internet"])
            payment = st.selectbox("Payment Method", sorted(df["Payment_Method"].unique()))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c3:
            monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 200.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charge * max(tenure,1)))
            total_revenue = st.number_input("Total Revenue ($)", 0.0, 10000.0, float(total_charges))
            value_deal = st.selectbox("Value Deal", ["No Deal"] + sorted([v for v in df["Value_Deal"].unique() if v != "No Deal"]))
            state = st.selectbox("State", sorted(df["State"].unique()))

        st.markdown("**Add-on services**")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        with a2:
            device_protect = st.selectbox("Device Protection", ["Yes", "No"])
            premium_support = st.selectbox("Premium Support", ["Yes", "No"])
        with a3:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        with a4:
            streaming_music = st.selectbox("Streaming Music", ["Yes", "No"])
            unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

        addon_count = sum(v == "Yes" for v in [online_security, online_backup, device_protect,
                                                 premium_support, streaming_tv, streaming_movies,
                                                 streaming_music, unlimited_data])
        input_data = {
            "Age": age, "Married": married, "Number_of_Referrals": referrals,
            "Tenure_in_Months": tenure, "Gender": gender, "Phone_Service": "Yes",
            "Multiple_Lines": multiple_lines, "Internet_Service": internet,
            "Internet_Type": internet_type, "Online_Security": online_security,
            "Online_Backup": online_backup, "Device_Protection_Plan": device_protect,
            "Premium_Support": premium_support, "Streaming_TV": streaming_tv,
            "Streaming_Movies": streaming_movies, "Streaming_Music": streaming_music,
            "Unlimited_Data": unlimited_data, "Contract": contract,
            "Paperless_Billing": paperless, "Payment_Method": payment,
            "Monthly_Charge": monthly_charge, "Total_Charges": total_charges,
            "Total_Refunds": 0.0, "Total_Extra_Data_Charges": 0,
            "Total_Long_Distance_Charges": 0.0, "Total_Revenue": total_revenue,
            "Total_Addon_Services": addon_count,
            "Avg_Monthly_Revenue": total_revenue / max(tenure, 1),
            "Extra_Charge_Ratio": 0.0, "Refund_Ratio": 0.0,
            "Is_Long_Term_Contract": int(contract != "Month-to-Month"),
            "Has_Value_Deal": int(value_deal != "No Deal"),
            "Is_High_Spender": int(monthly_charge >= df["Monthly_Charge"].quantile(0.75)),
            "Has_Referrals": int(referrals > 0), "Is_Senior": int(age >= 60),
            "Value_Deal": value_deal, "State": state,
        }

    if st.button("🔮 Predict Churn", type="primary"):
        X_input = encode_row(input_data, CHURN_FEATURES, CHURN_CATEGORICAL, churn_encoders, churn_scaler, CHURN_NUMERIC)
        proba = churn_model.predict_proba(X_input)[0, 1]
        pred = int(proba >= 0.5)

        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.subheader("Churn Prediction")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d62728" if proba >= 0.5 else "#2ca02c"},
                    "steps": [
                        {"range": [0, 30], "color": "#e6f4ea"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#f8d7da"},
                    ],
                },
            ))
            gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)

            if pred == 1:
                st.error(f"⚠️ **High churn risk** — predicted probability: {proba:.1%}")
            else:
                st.success(f"✅ **Likely to stay** — predicted churn probability: {proba:.1%}")

        with col2:
            st.subheader("🔍 Diagnostic Analysis — Why?")
            # Feature-importance-based diagnostic explanation
            top_feats = list(churn_importance.items())[:6]
            imp_df = pd.DataFrame(top_feats, columns=["Feature", "Importance"])
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Top global churn drivers (model feature importance)",
                         color="Importance", color_continuous_scale="Reds")
            fig.update_layout(height=280, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            # Business-rule diagnostic narrative tailored to this customer
            reasons = []
            if input_data["Contract"] == "Month-to-Month":
                reasons.append("No long-term contract — Month-to-Month customers churn far more often.")
            if input_data["Monthly_Charge"] >= df["Monthly_Charge"].quantile(0.75):
                reasons.append("Monthly charge is in the top 25% — price sensitivity risk.")
            if input_data["Tenure_in_Months"] < 12:
                reasons.append("Low tenure (< 12 months) — newer customers are more likely to leave.")
            if input_data["Total_Addon_Services"] <= 1:
                reasons.append("Very few add-on services — low engagement/stickiness.")
            if input_data.get("Number_of_Referrals", 0) == 0:
                reasons.append("No referrals made — weaker brand affinity signal.")
            if not reasons:
                reasons.append("No major single risk factor detected — profile looks healthy overall.")

            st.markdown("**Customer-specific risk factors:**")
            for r in reasons:
                st.markdown(f"- {r}")

        if pred == 1:
            st.subheader("📋 Predicted Churn Reason Category")
            X_reason = encode_row(input_data, CHURN_FEATURES, CHURN_CATEGORICAL, reason_encoders, reason_scaler, CHURN_NUMERIC)
            reason_pred = reason_model.predict(X_reason)[0]
            reason_proba = reason_model.predict_proba(X_reason)[0]
            reason_label = reason_le.inverse_transform([reason_pred])[0]
            conf = reason_proba.max()
            st.info(f"**Most likely churn driver category:** {reason_label}  (confidence: {conf:.1%})")
            st.write(CHURN_REASON_EXPLANATIONS.get(reason_label, "Review account details for a tailored save offer."))

            reason_prob_df = pd.DataFrame({
                "Category": reason_le.classes_,
                "Probability": reason_proba
            }).sort_values("Probability", ascending=False)
            fig2 = px.bar(reason_prob_df, x="Probability", y="Category", orientation="h",
                          title="Churn reason category probabilities", color="Probability",
                          color_continuous_scale="Oranges")
            fig2.update_layout(height=280, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📈 Model performance comparison (3 algorithms trained)"):
        res_df = pd.DataFrame(churn_results["results"]).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(res_df, use_container_width=True)
        st.caption(f"Best model selected by ROC-AUC: **{CHURN_META['best_model']}** (saved as best_churn_model.pkl)")
        fig3 = px.bar(res_df, x="Model", y="roc_auc", color="Model", title="ROC-AUC comparison across algorithms")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"Churn-reason diagnostic model accuracy (Random Forest, trained on churned subset): {reason_results['accuracy']:.2%}")

# =============================================================================
# SECTION 2: CUSTOMER CATEGORIZATION
# =============================================================================
elif section.startswith("2"):
    st.header("2️⃣ Customer Categorization")
    st.write(
        "Predicts which category a customer falls into based on their profile "
        "and behavior — used to prioritize retention and upsell efforts."
    )

    cat_descriptions = {
        "Premium-Ready": "🟢 High revenue + long-term contract. Great upsell candidate for premium bundles.",
        "Loyal": "🔵 Long tenure and/or strong add-on engagement. Low churn risk, reward with loyalty perks.",
        "Medium-Risk": "🟡 Some risk signals present (short tenure or month-to-month). Needs proactive engagement.",
        "High-Risk": "🔴 Strong churn signals or already churned profile pattern. Needs immediate retention action.",
    }

    mode2 = st.radio("Input mode", ["Pick existing customer", "Enter new customer profile"], horizontal=True, key="cat_mode")

    if mode2 == "Pick existing customer":
        cust_id2 = st.selectbox("Select Customer ID", df["Customer_ID"].sample(300, random_state=2).sort_values(), key="cat_cust")
        row2 = get_customer_row(cust_id2)
        input_data2 = row2.to_dict()
        st.subheader("Customer data")
        show_cols = ["Age", "Gender", "Married", "State", "Tenure_in_Months", "Contract",
                     "Monthly_Charge", "Total_Revenue", "Total_Addon_Services", "Number_of_Referrals"]
        st.dataframe(pd.DataFrame([{c: input_data2[c] for c in show_cols}]), use_container_width=True)
    else:
        st.info("Using the same manual-entry form as Section 1 — switch there, fill it in, then come back and reselect 'Pick existing customer' with a matching profile, or use the quick fields below.")
        c1, c2 = st.columns(2)
        with c1:
            age2 = st.number_input("Age", 18, 100, 35, key="c2age")
            tenure2 = st.number_input("Tenure (months)", 0, 100, 12, key="c2tenure")
            contract2 = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"], key="c2contract")
            monthly2 = st.number_input("Monthly Charge", 0.0, 200.0, 65.0, key="c2mc")
        with c2:
            referrals2 = st.number_input("Number of Referrals", 0, 20, 0, key="c2ref")
            addons2 = st.slider("Total Addon Services", 0, 8, 2, key="c2addons")
            total_rev2 = st.number_input("Total Revenue", 0.0, 10000.0, float(monthly2 * max(tenure2,1)), key="c2rev")
            state2 = st.selectbox("State", sorted(df["State"].unique()), key="c2state")

        input_data2 = {
            "Age": age2, "Married": "No", "Number_of_Referrals": referrals2,
            "Tenure_in_Months": tenure2, "Gender": "Male", "Phone_Service": "Yes",
            "Multiple_Lines": "No", "Internet_Service": "Yes", "Internet_Type": "Fiber Optic",
            "Online_Security": "No", "Online_Backup": "No", "Device_Protection_Plan": "No",
            "Premium_Support": "No", "Streaming_TV": "No", "Streaming_Movies": "No",
            "Streaming_Music": "No", "Unlimited_Data": "Yes", "Contract": contract2,
            "Paperless_Billing": "Yes", "Payment_Method": "Credit Card",
            "Monthly_Charge": monthly2, "Total_Charges": total_rev2, "Total_Revenue": total_rev2,
            "Total_Addon_Services": addons2,
            "Avg_Monthly_Revenue": total_rev2 / max(tenure2, 1),
            "Has_Value_Deal": 0, "Is_High_Spender": int(monthly2 >= df["Monthly_Charge"].quantile(0.75)),
            "Has_Referrals": int(referrals2 > 0), "Is_Senior": int(age2 >= 60),
            "Value_Deal": "No Deal", "State": state2,
        }

    if st.button("🏷️ Predict Customer Category", type="primary"):
        X_cat = encode_row(input_data2, CAT_FEATURES, CAT_CATEGORICAL, cat_encoders, cat_scaler, CAT_NUMERIC)
        cat_pred = cat_model.predict(X_cat)[0]
        cat_proba = cat_model.predict_proba(X_cat)[0]
        cat_label = cat_le.inverse_transform([cat_pred])[0]

        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.subheader("Predicted Category")
            st.markdown(f"## {cat_label}")
            st.write(cat_descriptions.get(cat_label, ""))
            conf = cat_proba.max()
            st.metric("Model Confidence", f"{conf:.1%}")

        with col2:
            proba_df = pd.DataFrame({"Category": cat_le.classes_, "Probability": cat_proba}).sort_values("Probability", ascending=False)
            fig = px.bar(proba_df, x="Probability", y="Category", orientation="h", color="Probability",
                         color_continuous_scale="Blues", title="Category probability breakdown")
            fig.update_layout(height=300, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Category distribution across the full customer base")
    dist = df["Customer_Category"].value_counts().reset_index()
    dist.columns = ["Category", "Count"]
    fig_dist = px.pie(dist, names="Category", values="Count", hole=0.45,
                       title="Customer Category Distribution", color="Category",
                       color_discrete_map={"Premium-Ready": "#2ca02c", "Loyal": "#1f77b4",
                                            "Medium-Risk": "#ff7f0e", "High-Risk": "#d62728"})
    st.plotly_chart(fig_dist, use_container_width=True)

    with st.expander("📈 Model performance comparison (3 algorithms trained)"):
        res_df2 = pd.DataFrame(cat_results["results"]).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(res_df2, use_container_width=True)
        st.caption(f"Best model selected by accuracy: **{CAT_META['best_model']}** (saved as best_category_model.pkl)")
        fig4 = px.bar(res_df2, x="Model", y="accuracy", color="Model", title="Accuracy comparison across algorithms")
        st.plotly_chart(fig4, use_container_width=True)

# =============================================================================
# SECTION 3: RECOMMENDATION ENGINE
# =============================================================================
elif section.startswith("3"):
    st.header("3️⃣ Subscription Recommendation Engine")
    st.write(
        "Recommends the best-fit subscription plan for a customer based on their "
        "primary complaint/need and profile — trained on a custom subscription "
        "plans catalogue using a single ML algorithm (Random Forest)."
    )

    complaints = sorted(plans_df["Primary_Complaint"].unique())
    c1, c2, c3 = st.columns(3)
    with c1:
        complaint = st.selectbox("Primary complaint / need", complaints)
        contract_pref = st.selectbox("Contract preference", sorted(plans_df["Contract_Preference"].unique()))
    with c2:
        charge_band = st.selectbox("Current monthly charge band", sorted(plans_df["Monthly_Charge_Band"].unique()))
        internet_need = st.selectbox("Internet type", sorted(plans_df["Internet_Type_Need"].unique()))
    with c3:
        family_size = st.selectbox("Household size", sorted(plans_df["Family_Size_Need"].unique()))

    if st.button("🎁 Get Recommendation", type="primary"):
        row = {
            "Primary_Complaint": complaint, "Contract_Preference": contract_pref,
            "Monthly_Charge_Band": charge_band, "Internet_Type_Need": internet_need,
            "Family_Size_Need": family_size,
        }
        X_rec = pd.DataFrame([row])
        for col in X_rec.columns:
            le = rec_encoders[col]
            val = str(X_rec.at[0, col])
            if val not in le.classes_:
                val = le.classes_[0]
            X_rec[col] = le.transform([val])

        plan_pred = rec_clf.predict(X_rec)[0]
        plan_label = rec_plan_le.inverse_transform([plan_pred])[0]
        price_pred = rec_reg.predict(X_rec)[0]

        plan_info = plans_df[plans_df["Recommended_Plan"] == plan_label].iloc[0]

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("🎯 Recommended Plan")
            st.markdown(f"## {plan_label}")
            st.write(plan_info["Plan_Description"])
            st.metric("Estimated Monthly Price", f"${plan_info['Recommended_Price']:.2f}")
            st.caption(f"Model-predicted price estimate: ${price_pred:.2f}")

        with col2:
            plan_proba = rec_clf.predict_proba(X_rec)[0]
            top_idx = np.argsort(plan_proba)[::-1][:5]
            top_plans = rec_plan_le.inverse_transform(top_idx)
            top_probs = plan_proba[top_idx]
            rec_df = pd.DataFrame({"Plan": top_plans, "Confidence": top_probs})
            fig = px.bar(rec_df, x="Confidence", y="Plan", orientation="h", color="Confidence",
                         color_continuous_scale="Greens", title="Top plan matches")
            fig.update_layout(height=300, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📚 Full Subscription Plans Catalogue")
    st.dataframe(plans_df, use_container_width=True)

    with st.expander("📈 Recommendation model evaluation (MSE, R², Chi-Squared)"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Classification Accuracy (5-fold CV)", f"{rec_results['classification_accuracy']:.1%}")
        m2.metric("Price Regression MSE", f"{rec_results['regression_mse']:.2f}")
        m3.metric("Price Regression R²", f"{rec_results['regression_r2']:.3f}")
        m4.metric("Chi-Squared statistic", f"{rec_results['chi_squared']:.1f}")
        st.caption(
            f"Algorithm: {rec_results['algorithm']}. Chi-Squared p-value = "
            f"{rec_results['chi_squared_p_value']:.4f} (dof={rec_results['chi_squared_dof']}) — "
            "tests whether Primary_Complaint and Recommended_Plan are statistically associated "
            "(low p-value confirms the complaint meaningfully drives the recommendation)."
        )
        st.write(
            "The recommendation engine uses **one ML algorithm (Random Forest)** in two modes: "
            "a classifier for the plan name and a regressor for the expected monthly price, "
            "evaluated with 5-fold cross-validation given the small (52-row) custom catalogue."
        )

# =============================================================================
# SECTION 4: INTERACTIVE DASHBOARD
# =============================================================================
elif section.startswith("4"):
    st.header("4️⃣ Interactive Business Intelligence Dashboard")

    # --- KPI row ---
    total_customers = len(df)
    churn_rate = (df["Customer_Status"] == "Churned").mean()
    avg_tenure = df["Tenure_in_Months"].mean()
    total_revenue = df["Total_Revenue"].sum()
    avg_monthly_charge = df["Monthly_Charge"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Customers", f"{total_customers:,}")
    k2.metric("Churn Rate", f"{churn_rate:.1%}")
    k3.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
    k4.metric("Total Revenue", f"${total_revenue:,.0f}")
    k5.metric("Avg Monthly Charge", f"${avg_monthly_charge:.2f}")

    st.markdown("---")

    # --- Filters ---
    with st.expander("🔧 Filters"):
        f1, f2, f3 = st.columns(3)
        with f1:
            state_filter = st.multiselect("State", sorted(df["State"].unique()))
        with f2:
            contract_filter = st.multiselect("Contract", sorted(df["Contract"].unique()))
        with f3:
            status_filter = st.multiselect("Customer Status", sorted(df["Customer_Status"].unique()))

    fdf = df.copy()
    if state_filter:
        fdf = fdf[fdf["State"].isin(state_filter)]
    if contract_filter:
        fdf = fdf[fdf["Contract"].isin(contract_filter)]
    if status_filter:
        fdf = fdf[fdf["Customer_Status"].isin(status_filter)]

    tab1, tab2, tab3, tab4 = st.tabs(["Churn Patterns", "Revenue & Tenure Trends", "Payment & Contract Insights", "Customer Segments"])

    # --- TAB 1: Churn Patterns ---
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            status_counts = fdf["Customer_Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig = px.bar(status_counts, x="Status", y="Count", color="Status",
                         title="Customer Status Distribution",
                         color_discrete_map={"Stayed": "#2ca02c", "Churned": "#d62728", "Joined": "#1f77b4"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            churned = fdf[fdf["Customer_Status"] == "Churned"]
            if len(churned) > 0:
                cat_counts = churned["Churn_Category"].value_counts().reset_index()
                cat_counts.columns = ["Churn_Category", "Count"]
                fig2 = px.pie(cat_counts, names="Churn_Category", values="Count", hole=0.4,
                              title="Churn Reason Category Breakdown")
                st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            churn_by_contract = fdf.groupby("Contract")["Churn_Label"].mean().reset_index()
            churn_by_contract.columns = ["Contract", "Churn_Rate"]
            fig3 = px.bar(churn_by_contract, x="Contract", y="Churn_Rate", color="Contract",
                          title="Churn Rate by Contract Type", text_auto=".1%")
            fig3.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            churn_by_tenure = fdf.groupby("Tenure_Bucket", observed=True)["Churn_Label"].mean().reset_index()
            churn_by_tenure.columns = ["Tenure_Bucket", "Churn_Rate"]
            fig4 = px.bar(churn_by_tenure, x="Tenure_Bucket", y="Churn_Rate", color="Tenure_Bucket",
                          title="Churn Rate by Tenure Bucket", text_auto=".1%")
            fig4.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig4, use_container_width=True)

        churn_by_state = fdf.groupby("State")["Churn_Label"].mean().reset_index().sort_values("Churn_Label", ascending=False)
        churn_by_state.columns = ["State", "Churn_Rate"]
        fig5 = px.bar(churn_by_state, x="State", y="Churn_Rate", title="Churn Rate by State", color="Churn_Rate",
                      color_continuous_scale="Reds")
        fig5.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig5, use_container_width=True)

    # --- TAB 2: Revenue & Tenure Trends ---
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig6 = px.histogram(fdf, x="Tenure_in_Months", nbins=30, title="Tenure Distribution",
                                color_discrete_sequence=["#1f77b4"])
            st.plotly_chart(fig6, use_container_width=True)
        with c2:
            fig7 = px.histogram(fdf, x="Total_Revenue", nbins=30, title="Total Revenue Distribution",
                                color_discrete_sequence=["#2ca02c"])
            st.plotly_chart(fig7, use_container_width=True)

        # Simulated monthly trend using tenure as a proxy for "months since acquisition"
        st.subheader("📅 Simulated Revenue Trend by Tenure Cohort")
        st.caption("The dataset has no explicit signup date, so tenure (months with company) is used as a time-based cohort proxy.")
        trend = fdf.groupby("Tenure_in_Months").agg(
            Avg_Revenue=("Total_Revenue", "mean"),
            Customer_Count=("Customer_ID", "count"),
            Churn_Rate=("Churn_Label", "mean")
        ).reset_index()
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=trend["Tenure_in_Months"], y=trend["Avg_Revenue"], mode="lines", name="Avg Revenue ($)"))
        fig8.update_layout(title="Average Revenue by Tenure (months)", xaxis_title="Tenure (months)", yaxis_title="Avg Revenue ($)")
        st.plotly_chart(fig8, use_container_width=True)

        fig9 = px.line(trend, x="Tenure_in_Months", y="Churn_Rate", title="Churn Rate Trend by Tenure (months)")
        fig9.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig9, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig10 = px.scatter(fdf.sample(min(1500, len(fdf))), x="Tenure_in_Months", y="Total_Revenue",
                               color="Customer_Status", title="Tenure vs Total Revenue",
                               color_discrete_map={"Stayed": "#2ca02c", "Churned": "#d62728", "Joined": "#1f77b4"})
            st.plotly_chart(fig10, use_container_width=True)
        with c4:
            addon_rev = fdf.groupby("Total_Addon_Services")["Total_Revenue"].mean().reset_index()
            fig11 = px.bar(addon_rev, x="Total_Addon_Services", y="Total_Revenue",
                           title="Avg Revenue by Number of Add-on Services")
            st.plotly_chart(fig11, use_container_width=True)

    # --- TAB 3: Payment & Contract Insights ---
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            payment_counts = fdf["Payment_Method"].value_counts().reset_index()
            payment_counts.columns = ["Payment_Method", "Count"]
            fig12 = px.pie(payment_counts, names="Payment_Method", values="Count", hole=0.4,
                          title="Payment Method Distribution")
            st.plotly_chart(fig12, use_container_width=True)
        with c2:
            churn_by_payment = fdf.groupby("Payment_Method")["Churn_Label"].mean().reset_index()
            churn_by_payment.columns = ["Payment_Method", "Churn_Rate"]
            fig13 = px.bar(churn_by_payment, x="Payment_Method", y="Churn_Rate", color="Payment_Method",
                           title="Churn Rate by Payment Method", text_auto=".1%")
            fig13.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig13, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            contract_counts = fdf["Contract"].value_counts().reset_index()
            contract_counts.columns = ["Contract", "Count"]
            fig14 = px.bar(contract_counts, x="Contract", y="Count", color="Contract", title="Contract Type Distribution")
            st.plotly_chart(fig14, use_container_width=True)
        with c4:
            internet_counts = fdf["Internet_Type"].value_counts().reset_index()
            internet_counts.columns = ["Internet_Type", "Count"]
            fig15 = px.bar(internet_counts, x="Internet_Type", y="Count", color="Internet_Type", title="Internet Type Distribution")
            st.plotly_chart(fig15, use_container_width=True)

        st.subheader("Add-on Service Adoption Rates")
        addon_cols = ["Online_Security", "Online_Backup", "Device_Protection_Plan", "Premium_Support",
                     "Streaming_TV", "Streaming_Movies", "Streaming_Music", "Unlimited_Data"]
        adoption = pd.DataFrame({
            "Service": addon_cols,
            "Adoption_Rate": [(fdf[c] == "Yes").mean() for c in addon_cols]
        }).sort_values("Adoption_Rate", ascending=True)
        fig16 = px.bar(adoption, x="Adoption_Rate", y="Service", orientation="h", title="Add-on Service Adoption Rates",
                       color="Adoption_Rate", color_continuous_scale="Purples")
        fig16.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig16, use_container_width=True)

    # --- TAB 4: Customer Segments ---
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            seg_counts = fdf["Customer_Category"].value_counts().reset_index()
            seg_counts.columns = ["Category", "Count"]
            fig17 = px.pie(seg_counts, names="Category", values="Count", hole=0.4, title="Customer Category Distribution",
                          color="Category", color_discrete_map={"Premium-Ready": "#2ca02c", "Loyal": "#1f77b4",
                                                                   "Medium-Risk": "#ff7f0e", "High-Risk": "#d62728"})
            st.plotly_chart(fig17, use_container_width=True)
        with c2:
            seg_rev = fdf.groupby("Customer_Category")["Total_Revenue"].mean().reset_index()
            fig18 = px.bar(seg_rev, x="Customer_Category", y="Total_Revenue", color="Customer_Category",
                           title="Average Revenue by Customer Category")
            st.plotly_chart(fig18, use_container_width=True)

        fig19 = px.box(fdf, x="Customer_Category", y="Tenure_in_Months", color="Customer_Category",
                       title="Tenure Distribution by Customer Category")
        st.plotly_chart(fig19, use_container_width=True)

        st.subheader("🔎 Explore Raw Data")
        st.dataframe(fdf.head(200), use_container_width=True)
        st.caption(f"Showing first 200 of {len(fdf):,} filtered rows.")

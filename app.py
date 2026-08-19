"""
CustomerRetention AI Streamlit web application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CustomerRetention AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE = Path(__file__).parent

DATA_PATH = BASE / "data" / "processed" / "customer_data_clean.csv"
PLANS_PATH = BASE / "data" / "subscription_plans.csv"
MODEL_DIR = BASE / "models"


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_plans():
    return pd.read_csv(PLANS_PATH)


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_churn_artifacts():

    model = joblib.load(MODEL_DIR / "best_churn_model.pkl")

    encoders = joblib.load(
        MODEL_DIR / "churn_encoders.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "churn_scaler.pkl"
    )

    with open(
        MODEL_DIR / "churn_model_results.json"
    ) as f:
        results = json.load(f)

    with open(
        MODEL_DIR / "churn_feature_importance.json"
    ) as f:
        importance = json.load(f)

    return model, encoders, scaler, results, importance


@st.cache_resource
def load_churn_reason_artifacts():

    model = joblib.load(
        MODEL_DIR / "churn_reason_model.pkl"
    )

    encoders = joblib.load(
        MODEL_DIR / "churn_reason_encoders.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "churn_reason_scaler.pkl"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "churn_reason_label_encoder.pkl"
    )

    with open(
        MODEL_DIR / "churn_reason_results.json"
    ) as f:
        results = json.load(f)

    return (
        model,
        encoders,
        scaler,
        label_encoder,
        results
    )


@st.cache_resource
def load_category_artifacts():

    model = joblib.load(
        MODEL_DIR / "best_category_model.pkl"
    )

    encoders = joblib.load(
        MODEL_DIR / "category_encoders.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "category_scaler.pkl"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "category_label_encoder.pkl"
    )

    with open(
        MODEL_DIR / "category_model_results.json"
    ) as f:
        results = json.load(f)

    return (
        model,
        encoders,
        scaler,
        label_encoder,
        results
    )


@st.cache_resource
def load_recommendation_artifacts():

    classifier = joblib.load(
        MODEL_DIR / "recommendation_model.pkl"
    )

    regressor = joblib.load(
        MODEL_DIR / "recommendation_price_model.pkl"
    )

    encoders = joblib.load(
        MODEL_DIR / "recommendation_encoders.pkl"
    )

    plan_encoder = joblib.load(
        MODEL_DIR / "recommendation_plan_label_encoder.pkl"
    )

    with open(
        MODEL_DIR / "recommendation_model_results.json"
    ) as f:
        results = json.load(f)

    return (
        classifier,
        regressor,
        encoders,
        plan_encoder,
        results
    )


# =============================================================================
# LOAD EVERYTHING
# =============================================================================

df = load_data()
plans_df = load_plans()


(
    churn_model,
    churn_encoders,
    churn_scaler,
    churn_results,
    churn_importance
) = load_churn_artifacts()


(
    reason_model,
    reason_encoders,
    reason_scaler,
    reason_le,
    reason_results
) = load_churn_reason_artifacts()


(
    cat_model,
    cat_encoders,
    cat_scaler,
    cat_le,
    cat_results
) = load_category_artifacts()


(
    rec_clf,
    rec_reg,
    rec_encoders,
    rec_plan_le,
    rec_results
) = load_recommendation_artifacts()


# =============================================================================
# LOAD MODEL METADATA
# =============================================================================

with open(
    MODEL_DIR / "churn_model_results.json"
) as f:
    CHURN_META = json.load(f)


with open(
    MODEL_DIR / "category_model_results.json"
) as f:
    CAT_META = json.load(f)


# Load churn reason metadata if present.
# If your churn reason model was trained using the same features as churn,
# this fallback will work.

REASON_META_PATH = MODEL_DIR / "churn_reason_results.json"

with open(REASON_META_PATH) as f:
    REASON_META = json.load(f)


CHURN_FEATURES = CHURN_META["features"]
CHURN_CATEGORICAL = CHURN_META["categorical"]
CHURN_NUMERIC = CHURN_META["numeric"]


CAT_FEATURES = CAT_META["features"]
CAT_CATEGORICAL = CAT_META["categorical"]
CAT_NUMERIC = CAT_META["numeric"]


# Churn reason model metadata.
# If the JSON contains feature information, use it.
# Otherwise use churn model features.

REASON_FEATURES = REASON_META.get(
    "features",
    CHURN_FEATURES
)

REASON_CATEGORICAL = REASON_META.get(
    "categorical",
    CHURN_CATEGORICAL
)

REASON_NUMERIC = REASON_META.get(
    "numeric",
    CHURN_NUMERIC
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def encode_row(
    row_dict,
    feature_list,
    categorical_list,
    encoders,
    scaler,
    numeric_list
):

    X = pd.DataFrame([row_dict])

    X = X[feature_list].copy()

    for col in categorical_list:

        le = encoders[col]

        value = str(X.at[0, col])

        if value not in le.classes_:

            # Safe fallback
            value = le.classes_[0]

        X[col] = le.transform([value])

    if numeric_list:
        X[numeric_list] = scaler.transform(
            X[numeric_list]
        )

    return X


def get_customer_row(customer_id):

    row = df[
        df["Customer_ID"] == customer_id
    ]

    if row.empty:
        return None

    return row.iloc[0]


# =============================================================================
# CHURN REASON EXPLANATIONS
# =============================================================================

CHURN_REASON_EXPLANATIONS = {

    "Competitor":
        "This customer profile resembles customers who left for a competitor offering better devices, pricing, or data plans.",

    "Attitude":
        "Service or support interaction quality appears to be the main risk factor.",

    "Dissatisfaction":
        "General dissatisfaction with service reliability, network quality, or products appears to be the dominant risk factor.",

    "Price":
        "Price sensitivity appears to be the leading risk factor. Monthly charges may be high relative to perceived value.",

    "Other":
        "Risk factors are mixed and are not dominated by a single category. A general retention outreach is recommended."
}


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>

    .metric-card {
        background-color: #f5f3fb;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #eaefe5;
    }

    .big-font {
        font-size: 22px !important;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# APP HEADER
# =============================================================================

st.title("CustomerRetention AI")

st.caption(
    "Churn Prediction • Diagnostic Root-Cause Analysis • "
    "Customer Categorization • Subscription Recommendation • "
    "Interactive Dashboard"
)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.header("Navigation")

    section = st.radio(
        "Choose a section",
        [
            "1️ Churn Prediction & Diagnosis",
            "2️ Customer Categorization",
            "3️ Recommendation Engine",
            "4️ Interactive Dashboard"
        ]
    )

    st.markdown("---")

    st.markdown("### Project Description")

    st.markdown(
        "CustomerRetention AI is an end-to-end machine learning project "
        "covering business understanding, data cleaning, feature engineering, "
        "EDA, multi-model training and selection, and deployment using Streamlit."
    )

    st.markdown(
        f"**Dataset size:** {len(df):,} customers"
    )

    best_churn = CHURN_META["best_model"]

    churn_auc = churn_results[
        "results"
    ][best_churn]["roc_auc"]

    st.markdown(
        f"**Churn model:** {best_churn} "
        f"(ROC-AUC {churn_auc})"
    )

    best_category = CAT_META["best_model"]

    category_accuracy = cat_results[
        "results"
    ][best_category]["accuracy"]

    st.markdown(
        f"**Category model:** {best_category} "
        f"(Accuracy {category_accuracy})"
    )


# =============================================================================
# SECTION 1 — CHURN PREDICTION
# =============================================================================

if section.startswith("1"):

    st.header(
        "1. Churn Prediction & Diagnostic Analysis"
    )

    st.write(
        "Select an existing customer or enter a new customer profile "
        "to predict churn probability and identify the major risk factors."
    )


    mode = st.radio(
        "Input mode",
        [
            "Pick existing customer",
            "Enter new customer profile"
        ],
        horizontal=True
    )


    # -------------------------------------------------------------------------
    # EXISTING CUSTOMER
    # -------------------------------------------------------------------------

    if mode == "Pick existing customer":

        cust_id = st.selectbox(
            "Select Customer ID",
            df["Customer_ID"]
            .sample(
                min(300, len(df)),
                random_state=1
            )
            .sort_values()
        )

        row = get_customer_row(cust_id)

        input_data = row.to_dict()


    # -------------------------------------------------------------------------
    # MANUAL CUSTOMER INPUT
    # -------------------------------------------------------------------------

    else:

        st.subheader("Enter Customer Profile")

        c1, c2, c3 = st.columns(3)


        with c1:

            age = st.number_input(
                "Age",
                18,
                100,
                35
            )

            married = st.selectbox(
                "Married",
                sorted(df["Married"].astype(str).unique())
            )

            tenure = st.number_input(
                "Tenure (months)",
                0,
                100,
                12
            )

            referrals = st.number_input(
                "Number of Referrals",
                0,
                20,
                0
            )

            gender = st.selectbox(
                "Gender",
                sorted(df["Gender"].astype(str).unique())
            )


        with c2:

            contract = st.selectbox(
                "Contract",
                sorted(df["Contract"].astype(str).unique())
            )

            internet = st.selectbox(
                "Internet Service",
                sorted(df["Internet_Service"].astype(str).unique())
            )

            internet_type = st.selectbox(
                "Internet Type",
                sorted(df["Internet_Type"].astype(str).unique())
            )

            payment = st.selectbox(
                "Payment Method",
                sorted(
                    df["Payment_Method"]
                    .astype(str)
                    .unique()
                )
            )

            paperless = st.selectbox(
                "Paperless Billing",
                sorted(
                    df["Paperless_Billing"]
                    .astype(str)
                    .unique()
                )
            )


        with c3:

            monthly_charge = st.number_input(
                "Monthly Charge ($)",
                0.0,
                200.0,
                65.0
            )

            total_charges = st.number_input(
                "Total Charges ($)",
                0.0,
                10000.0,
                float(
                    monthly_charge *
                    max(tenure, 1)
                )
            )

            total_revenue = st.number_input(
                "Total Revenue ($)",
                0.0,
                10000.0,
                float(total_charges)
            )

            value_deal = st.selectbox(
                "Value Deal",
                sorted(
                    df["Value_Deal"]
                    .astype(str)
                    .unique()
                )
            )

            state = st.selectbox(
                "State",
                sorted(
                    df["State"]
                    .astype(str)
                    .unique()
                )
            )


        st.markdown(
            "**Add-on Services**"
        )

        a1, a2, a3, a4 = st.columns(4)


        with a1:

            online_security = st.selectbox(
                "Online Security",
                sorted(
                    df["Online_Security"]
                    .astype(str)
                    .unique()
                )
            )

            online_backup = st.selectbox(
                "Online Backup",
                sorted(
                    df["Online_Backup"]
                    .astype(str)
                    .unique()
                )
            )


        with a2:

            device_protect = st.selectbox(
                "Device Protection",
                sorted(
                    df["Device_Protection_Plan"]
                    .astype(str)
                    .unique()
                )
            )

            premium_support = st.selectbox(
                "Premium Support",
                sorted(
                    df["Premium_Support"]
                    .astype(str)
                    .unique()
                )
            )


        with a3:

            streaming_tv = st.selectbox(
                "Streaming TV",
                sorted(
                    df["Streaming_TV"]
                    .astype(str)
                    .unique()
                )
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                sorted(
                    df["Streaming_Movies"]
                    .astype(str)
                    .unique()
                )
            )


        with a4:

            streaming_music = st.selectbox(
                "Streaming Music",
                sorted(
                    df["Streaming_Music"]
                    .astype(str)
                    .unique()
                )
            )

            unlimited_data = st.selectbox(
                "Unlimited Data",
                sorted(
                    df["Unlimited_Data"]
                    .astype(str)
                    .unique()
                )
            )


        multiple_lines = st.selectbox(
            "Multiple Lines",
            sorted(
                df["Multiple_Lines"]
                .astype(str)
                .unique()
            )
        )


        addon_count = sum(

            value == "Yes"

            for value in [

                online_security,
                online_backup,
                device_protect,
                premium_support,
                streaming_tv,
                streaming_movies,
                streaming_music,
                unlimited_data

            ]
        )


        input_data = {

            "Age": age,

            "Married": married,

            "Number_of_Referrals": referrals,

            "Tenure_in_Months": tenure,

            "Gender": gender,

            "Phone_Service": "Yes",

            "Multiple_Lines": multiple_lines,

            "Internet_Service": internet,

            "Internet_Type": internet_type,

            "Online_Security": online_security,

            "Online_Backup": online_backup,

            "Device_Protection_Plan": device_protect,

            "Premium_Support": premium_support,

            "Streaming_TV": streaming_tv,

            "Streaming_Movies": streaming_movies,

            "Streaming_Music": streaming_music,

            "Unlimited_Data": unlimited_data,

            "Contract": contract,

            "Paperless_Billing": paperless,

            "Payment_Method": payment,

            "Monthly_Charge": monthly_charge,

            "Total_Charges": total_charges,

            "Total_Refunds": 0.0,

            "Total_Extra_Data_Charges": 0.0,

            "Total_Long_Distance_Charges": 0.0,

            "Total_Revenue": total_revenue,

            "Total_Addon_Services": addon_count,

            "Avg_Monthly_Revenue":
                total_revenue / max(tenure, 1),

            "Extra_Charge_Ratio": 0.0,

            "Refund_Ratio": 0.0,

            "Is_Long_Term_Contract":
                int(contract != "Month-to-Month"),

            "Has_Value_Deal":
                int(value_deal != "No Deal"),

            "Is_High_Spender":
                int(
                    monthly_charge >=
                    df["Monthly_Charge"].quantile(0.75)
                ),

            "Has_Referrals":
                int(referrals > 0),

            "Is_Senior":
                int(age >= 60),

            "Value_Deal": value_deal,

            "State": state
        }


    # -------------------------------------------------------------------------
    # PREDICT CHURN
    # -------------------------------------------------------------------------

    if st.button(
        "Predict Churn",
        type="primary"
    ):

        X_input = encode_row(
            input_data,
            CHURN_FEATURES,
            CHURN_CATEGORICAL,
            churn_encoders,
            churn_scaler,
            CHURN_NUMERIC
        )


        probabilities = churn_model.predict_proba(
            X_input
        )[0]


        # Find probability corresponding to churn class = 1
        churn_class_index = list(
            churn_model.classes_
        ).index(1)


        proba = probabilities[
            churn_class_index
        ]


        pred = int(
            proba >= 0.5
        )


        col1, col2 = st.columns(
            [1, 1.4]
        )


        with col1:

            st.subheader(
                "Churn Prediction"
            )


            gauge = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=proba * 100,

                    title={
                        "text":
                        "Churn Probability (%)"
                    },

                    gauge={

                        "axis": {
                            "range": [0, 100]
                        },

                        "steps": [

                            {
                                "range": [0, 30],
                                "color": "#c7efd2"
                            },

                            {
                                "range": [30, 60],
                                "color": "#f0e0ab"
                            },

                            {
                                "range": [60, 100],
                                "color": "#f1b2b7"
                            }

                        ]

                    }

                )

            )


            gauge.update_layout(
                height=280,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=10
                )
            )


            st.plotly_chart(
                gauge,
                use_container_width=True
            )


            if pred == 1:

                st.error(
                    f"High churn risk — "
                    f"predicted probability: "
                    f"{proba:.1%}"
                )

            else:

                st.success(
                    f"Likely to stay — "
                    f"predicted churn probability: "
                    f"{proba:.1%}"
                )


        with col2:

            st.subheader(
                "Diagnostic Analysis — Why?"
            )


            top_feats = list(
                churn_importance.items()
            )[:6]


            imp_df = pd.DataFrame(
                top_feats,
                columns=[
                    "Feature",
                    "Importance"
                ]
            )


            fig = px.bar(

                imp_df,

                x="Importance",

                y="Feature",

                orientation="h",

                title=
                "Top Global Churn Drivers",

                color="Importance",

                color_continuous_scale="Reds"

            )


            fig.update_layout(
                height=280,

                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            reasons = []


            if (
                input_data["Contract"]
                == "Month-to-Month"
            ):

                reasons.append(
                    "Month-to-Month contract — "
                    "customers without long-term commitments "
                    "may have a higher risk of leaving."
                )


            if (
                input_data["Monthly_Charge"]
                >= df["Monthly_Charge"]
                .quantile(0.75)
            ):

                reasons.append(
                    "Monthly charge is in the top 25% "
                    "of the customer base."
                )


            if (
                input_data["Tenure_in_Months"]
                < 12
            ):

                reasons.append(
                    "Low tenure — newer customers "
                    "may be more likely to leave."
                )


            if (
                input_data["Total_Addon_Services"]
                <= 1
            ):

                reasons.append(
                    "Very few add-on services — "
                    "lower service engagement."
                )


            if (
                input_data.get(
                    "Number_of_Referrals",
                    0
                ) == 0
            ):

                reasons.append(
                    "No referrals recorded — "
                    "weaker customer engagement signal."
                )


            if not reasons:

                reasons.append(
                    "No major individual risk factor "
                    "was detected."
                )


            st.markdown(
                "**Customer-specific risk factors:**"
            )


            for reason in reasons:

                st.markdown(
                    f"- {reason}"
                )


        # ---------------------------------------------------------------------
        # CHURN REASON MODEL
        # ---------------------------------------------------------------------

        if pred == 1:

            st.subheader(
                "Predicted Churn Reason Category"
            )


            X_reason = encode_row(

                input_data,

                REASON_FEATURES,

                REASON_CATEGORICAL,

                reason_encoders,

                reason_scaler,

                REASON_NUMERIC

            )


            reason_pred = reason_model.predict(
                X_reason
            )[0]


            reason_proba = reason_model.predict_proba(
                X_reason
            )[0]


            reason_label = reason_le.inverse_transform(
                [reason_pred]
            )[0]


            confidence = reason_proba.max()


            st.info(

                f"Most likely churn driver category: "
                f"{reason_label} "
                f"(confidence: {confidence:.1%})"

            )


            st.write(

                CHURN_REASON_EXPLANATIONS.get(

                    reason_label,

                    "Review account details for a "
                    "tailored retention offer."

                )

            )


            reason_prob_df = pd.DataFrame({

                "Category":
                    reason_le.classes_,

                "Probability":
                    reason_proba

            }).sort_values(

                "Probability",

                ascending=False

            )


            fig2 = px.bar(

                reason_prob_df,

                x="Probability",

                y="Category",

                orientation="h",

                title=
                "Churn Reason Category Probabilities",

                color="Probability",

                color_continuous_scale="Oranges"

            )


            fig2.update_layout(

                height=280,

                yaxis={
                    "categoryorder":
                    "total ascending"
                }

            )


            st.plotly_chart(

                fig2,

                use_container_width=True

            )


    # -------------------------------------------------------------------------
    # CHURN MODEL PERFORMANCE
    # -------------------------------------------------------------------------

    with st.expander(
        "Model Performance Comparison "
        "(3 Algorithms Trained)"
    ):

        res_df = (

            pd.DataFrame(
                churn_results["results"]
            )

            .T

            .reset_index()

            .rename(
                columns={
                    "index":
                    "Model"
                }
            )

        )


        st.dataframe(
            res_df,
            use_container_width=True
        )


        st.caption(

            f"Best model selected by ROC-AUC: "
            f"**{CHURN_META['best_model']}**"

        )


        fig3 = px.bar(

            res_df,

            x="Model",

            y="roc_auc",

            color="Model",

            title=
            "ROC-AUC Comparison Across Algorithms"

        )


        st.plotly_chart(
            fig3,
            use_container_width=True
        )


# =============================================================================
# SECTION 2 — CUSTOMER CATEGORIZATION
# =============================================================================

elif section.startswith("2"):

    st.header(
        "2. Customer Categorization"
    )


    st.write(
        "Predicts which category a customer falls into "
        "based on their profile and behavior."
    )


    cat_descriptions = {

        "Premium-Ready":
            "** High revenue and strong long-term value. "
            "Potential candidate for premium bundles.",

        "Loyal":
            "## Strong tenure or service engagement. "
            "Low-risk customer suitable for loyalty benefits.",

        "Medium-Risk":
            "$$ Some risk signals are present. "
            "Proactive engagement may help retention.",

        "High-Risk":
            "&& Strong churn-related risk signals. "
            "Immediate retention action is recommended."

    }


    mode2 = st.radio(

        "Input mode",

        [
            "Pick existing customer",
            "Enter new customer profile"
        ],

        horizontal=True,

        key="cat_mode"

    )


    if mode2 == "Pick existing customer":

        cust_id2 = st.selectbox(

            "Select Customer ID",

            df["Customer_ID"]
            .sample(
                min(300, len(df)),
                random_state=2
            )
            .sort_values(),

            key="cat_cust"

        )


        row2 = get_customer_row(
            cust_id2
        )


        input_data2 = row2.to_dict()


        st.subheader(
            "Customer Data"
        )


        show_cols = [

            "Age",

            "Gender",

            "Married",

            "State",

            "Tenure_in_Months",

            "Contract",

            "Monthly_Charge",

            "Total_Revenue",

            "Total_Addon_Services",

            "Number_of_Referrals"

        ]


        st.dataframe(

            pd.DataFrame([{

                column:
                    input_data2[column]

                for column
                in show_cols

            }]),

            use_container_width=True

        )


    else:

        c1, c2 = st.columns(2)


        with c1:

            age2 = st.number_input(
                "Age",
                18,
                100,
                35,
                key="c2age"
            )

            tenure2 = st.number_input(
                "Tenure (months)",
                0,
                100,
                12,
                key="c2tenure"
            )

            contract2 = st.selectbox(
                "Contract",
                sorted(
                    df["Contract"]
                    .astype(str)
                    .unique()
                ),
                key="c2contract"
            )

            monthly2 = st.number_input(
                "Monthly Charge",
                0.0,
                200.0,
                65.0,
                key="c2mc"
            )


        with c2:

            referrals2 = st.number_input(
                "Number of Referrals",
                0,
                20,
                0,
                key="c2ref"
            )

            addons2 = st.slider(
                "Total Add-on Services",
                0,
                8,
                2,
                key="c2addons"
            )

            total_rev2 = st.number_input(
                "Total Revenue",
                0.0,
                10000.0,
                float(
                    monthly2 *
                    max(tenure2, 1)
                ),
                key="c2rev"
            )

            state2 = st.selectbox(
                "State",
                sorted(
                    df["State"]
                    .astype(str)
                    .unique()
                ),
                key="c2state"
            )


        input_data2 = {

            "Age": age2,

            "Married":
                df["Married"].astype(str).iloc[0],

            "Number_of_Referrals":
                referrals2,

            "Tenure_in_Months":
                tenure2,

            "Gender":
                df["Gender"].astype(str).iloc[0],

            "Phone_Service":
                df["Phone_Service"].astype(str).iloc[0],

            "Multiple_Lines":
                df["Multiple_Lines"].astype(str).iloc[0],

            "Internet_Service":
                df["Internet_Service"].astype(str).iloc[0],

            "Internet_Type":
                df["Internet_Type"].astype(str).iloc[0],

            "Online_Security":
                df["Online_Security"].astype(str).iloc[0],

            "Online_Backup":
                df["Online_Backup"].astype(str).iloc[0],

            "Device_Protection_Plan":
                df["Device_Protection_Plan"].astype(str).iloc[0],

            "Premium_Support":
                df["Premium_Support"].astype(str).iloc[0],

            "Streaming_TV":
                df["Streaming_TV"].astype(str).iloc[0],

            "Streaming_Movies":
                df["Streaming_Movies"].astype(str).iloc[0],

            "Streaming_Music":
                df["Streaming_Music"].astype(str).iloc[0],

            "Unlimited_Data":
                df["Unlimited_Data"].astype(str).iloc[0],

            "Contract":
                contract2,

            "Paperless_Billing":
                df["Paperless_Billing"]
                .astype(str)
                .iloc[0],

            "Payment_Method":
                df["Payment_Method"]
                .astype(str)
                .iloc[0],

            "Monthly_Charge":
                monthly2,

            "Total_Charges":
                total_rev2,

            "Total_Revenue":
                total_rev2,

            "Total_Addon_Services":
                addons2,

            "Avg_Monthly_Revenue":
                total_rev2 /
                max(tenure2, 1),

            "Has_Value_Deal":
                0,

            "Is_High_Spender":
                int(
                    monthly2 >=
                    df["Monthly_Charge"]
                    .quantile(0.75)
                ),

            "Has_Referrals":
                int(referrals2 > 0),

            "Is_Senior":
                int(age2 >= 60),

            "Value_Deal":
                "No Deal",

            "State":
                state2

        }


    if st.button(
        "Predict Customer Category",
        type="primary"
    ):

        X_cat = encode_row(

            input_data2,

            CAT_FEATURES,

            CAT_CATEGORICAL,

            cat_encoders,

            cat_scaler,

            CAT_NUMERIC

        )


        cat_pred = cat_model.predict(
            X_cat
        )[0]


        cat_proba = cat_model.predict_proba(
            X_cat
        )[0]


        cat_label = cat_le.inverse_transform(
            [cat_pred]
        )[0]


        col1, col2 = st.columns(
            [1, 1.3]
        )


        with col1:

            st.subheader(
                "Predicted Category"
            )

            st.markdown(
                f"## {cat_label}"
            )

            st.write(
                cat_descriptions.get(
                    cat_label,
                    ""
                )
            )

            confidence = cat_proba.max()

            st.metric(
                "Model Confidence",
                f"{confidence:.1%}"
            )


        with col2:

            proba_df = pd.DataFrame({

                "Category":
                    cat_le.classes_,

                "Probability":
                    cat_proba

            }).sort_values(

                "Probability",

                ascending=False

            )


            fig = px.bar(

                proba_df,

                x="Probability",

                y="Category",

                orientation="h",

                color="Probability",

                color_continuous_scale="Blues",

                title=
                "Category Probability Breakdown"

            )


            fig.update_layout(
                height=300
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    st.markdown("---")


    st.subheader(
        "Category Distribution Across the Full Customer Base"
    )


    dist = (
        df["Customer_Category"]
        .value_counts()
        .reset_index()
    )


    dist.columns = [
        "Category",
        "Count"
    ]


    fig_dist = px.pie(

        dist,

        names="Category",

        values="Count",

        hole=0.45,

        title=
        "Customer Category Distribution"

    )


    st.plotly_chart(
        fig_dist,
        use_container_width=True
    )


    with st.expander(
        "Model Performance Comparison "
        "(3 Algorithms Trained)"
    ):

        res_df2 = (

            pd.DataFrame(
                cat_results["results"]
            )

            .T

            .reset_index()

            .rename(
                columns={
                    "index":
                    "Model"
                }
            )

        )


        st.dataframe(
            res_df2,
            use_container_width=True
        )


        st.caption(

            f"Best model selected by accuracy: "
            f"**{CAT_META['best_model']}**"

        )


        fig4 = px.bar(

            res_df2,

            x="Model",

            y="accuracy",

            color="Model",

            title=
            "Accuracy Comparison Across Algorithms"

        )


        st.plotly_chart(
            fig4,
            use_container_width=True
        )


# =============================================================================
# SECTION 3 — RECOMMENDATION ENGINE
# =============================================================================

elif section.startswith("3"):

    st.header(
        "3. Subscription Recommendation Engine"
    )


    st.write(
        "Recommends a subscription plan based on "
        "customer needs and preferences."
    )


    complaints = sorted(
        plans_df[
            "Primary_Complaint"
        ]
        .astype(str)
        .unique()
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        complaint = st.selectbox(
            "Primary Complaint / Need",
            complaints
        )

        contract_pref = st.selectbox(
            "Contract Preference",
            sorted(
                plans_df[
                    "Contract_Preference"
                ]
                .astype(str)
                .unique()
            )
        )


    with c2:

        charge_band = st.selectbox(
            "Current Monthly Charge Band",
            sorted(
                plans_df[
                    "Monthly_Charge_Band"
                ]
                .astype(str)
                .unique()
            )
        )

        internet_need = st.selectbox(
            "Internet Type",
            sorted(
                plans_df[
                    "Internet_Type_Need"
                ]
                .astype(str)
                .unique()
            )
        )


    with c3:

        family_size = st.selectbox(
            "Household Size",
            sorted(
                plans_df[
                    "Family_Size_Need"
                ]
                .astype(str)
                .unique()
            )
        )


    if st.button(
        "Get Recommendation",
        type="primary"
    ):

        row = {

            "Primary_Complaint":
                complaint,

            "Contract_Preference":
                contract_pref,

            "Monthly_Charge_Band":
                charge_band,

            "Internet_Type_Need":
                internet_need,

            "Family_Size_Need":
                family_size

        }


        X_rec = pd.DataFrame(
            [row]
        )


        for col in X_rec.columns:

            le = rec_encoders[col]

            value = str(
                X_rec.at[0, col]
            )

            if value not in le.classes_:

                value = le.classes_[0]

            X_rec[col] = le.transform(
                [value]
            )


        plan_pred = rec_clf.predict(
            X_rec
        )[0]


        plan_label = rec_plan_le.inverse_transform(
            [plan_pred]
        )[0]


        price_pred = rec_reg.predict(
            X_rec
        )[0]


        plan_info = plans_df[
            plans_df[
                "Recommended_Plan"
            ]
            == plan_label
        ].iloc[0]


        col1, col2 = st.columns(
            [1.2, 1]
        )


        with col1:

            st.subheader(
                "Recommended Plan"
            )

            st.markdown(
                f"## {plan_label}"
            )

            st.write(
                plan_info[
                    "Plan_Description"
                ]
            )

            st.metric(
                "Estimated Monthly Price",
                f"${plan_info['Recommended_Price']:.2f}"
            )

            st.caption(
                f"Model-predicted price estimate: "
                f"${price_pred:.2f}"
            )


        with col2:

            plan_proba = rec_clf.predict_proba(
                X_rec
            )[0]


            top_idx = np.argsort(
                plan_proba
            )[::-1][:5]


            top_plans = rec_plan_le.inverse_transform(
                top_idx
            )


            top_probs = plan_proba[
                top_idx
            ]


            rec_df = pd.DataFrame({

                "Plan":
                    top_plans,

                "Confidence":
                    top_probs

            })


            fig = px.bar(

                rec_df,

                x="Confidence",

                y="Plan",

                orientation="h",

                color="Confidence",

                color_continuous_scale="Greens",

                title=
                "Top Plan Matches"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    st.markdown("---")


    st.subheader(
        "Full Subscription Plans Catalogue"
    )


    st.dataframe(
        plans_df,
        use_container_width=True
    )


# =============================================================================
# SECTION 4 — INTERACTIVE DASHBOARD
# =============================================================================

elif section.startswith("4"):

    st.header(
        "4. Interactive Business Intelligence Dashboard"
    )


    total_customers = len(df)

    churn_rate = (
        df["Customer_Status"]
        .astype(str)
        .eq("Churned")
        .mean()
    )

    avg_tenure = df[
        "Tenure_in_Months"
    ].mean()

    total_revenue = df[
        "Total_Revenue"
    ].sum()

    avg_monthly_charge = df[
        "Monthly_Charge"
    ].mean()


    k1, k2, k3, k4, k5 = st.columns(5)


    k1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    k2.metric(
        "Churn Rate",
        f"{churn_rate:.1%}"
    )

    k3.metric(
        "Avg Tenure (Months)",
        f"{avg_tenure:.1f}"
    )

    k4.metric(
        "Total Revenue",
        f"${total_revenue:,.0f}"
    )

    k5.metric(
        "Avg Monthly Charge",
        f"${avg_monthly_charge:.2f}"
    )


    st.markdown("---")


    with st.expander("Filters"):

        f1, f2, f3 = st.columns(3)


        with f1:

            state_filter = st.multiselect(
                "State",
                sorted(
                    df["State"]
                    .astype(str)
                    .unique()
                )
            )


        with f2:

            contract_filter = st.multiselect(
                "Contract",
                sorted(
                    df["Contract"]
                    .astype(str)
                    .unique()
                )
            )


        with f3:

            status_filter = st.multiselect(
                "Customer Status",
                sorted(
                    df["Customer_Status"]
                    .astype(str)
                    .unique()
                )
            )


    fdf = df.copy()


    if state_filter:

        fdf = fdf[
            fdf["State"]
            .astype(str)
            .isin(state_filter)
        ]


    if contract_filter:

        fdf = fdf[
            fdf["Contract"]
            .astype(str)
            .isin(contract_filter)
        ]


    if status_filter:

        fdf = fdf[
            fdf["Customer_Status"]
            .astype(str)
            .isin(status_filter)
        ]


    tab1, tab2, tab3, tab4 = st.tabs([

        "Churn Patterns",

        "Revenue & Tenure Trends",

        "Payment & Contract Insights",

        "Customer Segments"

    ])


    # =========================================================================
    # TAB 1
    # =========================================================================

    with tab1:

        status_counts = (
            fdf["Customer_Status"]
            .value_counts()
            .reset_index()
        )

        status_counts.columns = [
            "Status",
            "Count"
        ]


        fig = px.bar(

            status_counts,

            x="Status",

            y="Count",

            color="Status",

            title=
            "Customer Status Distribution"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        churn_by_contract = (

            fdf.groupby(
                "Contract"
            )["Churn_Label"]

            .mean()

            .reset_index()

        )


        churn_by_contract.columns = [

            "Contract",

            "Churn_Rate"

        ]


        fig2 = px.bar(

            churn_by_contract,

            x="Contract",

            y="Churn_Rate",

            color="Contract",

            title=
            "Churn Rate by Contract Type"

        )


        fig2.update_yaxes(
            tickformat=".0%"
        )


        st.plotly_chart(
            fig2,
            use_container_width=True
        )


        churn_by_tenure = (

            fdf.groupby(
                "Tenure_Bucket",
                observed=True
            )["Churn_Label"]

            .mean()

            .reset_index()

        )


        churn_by_tenure.columns = [

            "Tenure_Bucket",

            "Churn_Rate"

        ]


        fig3 = px.bar(

            churn_by_tenure,

            x="Tenure_Bucket",

            y="Churn_Rate",

            title=
            "Churn Rate by Tenure Bucket"

        )


        fig3.update_yaxes(
            tickformat=".0%"
        )


        st.plotly_chart(
            fig3,
            use_container_width=True
        )


    # =========================================================================
    # TAB 2
    # =========================================================================

    with tab2:

        fig4 = px.histogram(

            fdf,

            x="Tenure_in_Months",

            nbins=30,

            title=
            "Tenure Distribution"

        )


        st.plotly_chart(
            fig4,
            use_container_width=True
        )


        fig5 = px.histogram(

            fdf,

            x="Total_Revenue",

            nbins=30,

            title=
            "Total Revenue Distribution"

        )


        st.plotly_chart(
            fig5,
            use_container_width=True
        )


        trend = (

            fdf.groupby(
                "Tenure_in_Months"
            )

            .agg(

                Avg_Revenue=(
                    "Total_Revenue",
                    "mean"
                ),

                Customer_Count=(
                    "Customer_ID",
                    "count"
                ),

                Churn_Rate=(
                    "Churn_Label",
                    "mean"
                )

            )

            .reset_index()

        )


        fig6 = px.line(

            trend,

            x="Tenure_in_Months",

            y="Avg_Revenue",

            title=
            "Average Revenue by Tenure"

        )


        st.plotly_chart(
            fig6,
            use_container_width=True
        )


        fig7 = px.line(

            trend,

            x="Tenure_in_Months",

            y="Churn_Rate",

            title=
            "Churn Rate Trend by Tenure"

        )


        fig7.update_yaxes(
            tickformat=".0%"
        )


        st.plotly_chart(
            fig7,
            use_container_width=True
        )


    # =========================================================================
    # TAB 3
    # =========================================================================

    with tab3:

        payment_counts = (

            fdf[
                "Payment_Method"
            ]

            .value_counts()

            .reset_index()

        )


        payment_counts.columns = [

            "Payment_Method",

            "Count"

        ]


        fig8 = px.pie(

            payment_counts,

            names=
            "Payment_Method",

            values=
            "Count",

            hole=0.4,

            title=
            "Payment Method Distribution"

        )


        st.plotly_chart(
            fig8,
            use_container_width=True
        )


        churn_by_payment = (

            fdf.groupby(
                "Payment_Method"
            )["Churn_Label"]

            .mean()

            .reset_index()

        )


        churn_by_payment.columns = [

            "Payment_Method",

            "Churn_Rate"

        ]


        fig9 = px.bar(

            churn_by_payment,

            x=
            "Payment_Method",

            y=
            "Churn_Rate",

            color=
            "Payment_Method",

            title=
            "Churn Rate by Payment Method"

        )


        fig9.update_yaxes(
            tickformat=".0%"
        )


        st.plotly_chart(
            fig9,
            use_container_width=True
        )


    # =========================================================================
    # TAB 4
    # =========================================================================

    with tab4:

        seg_counts = (

            fdf[
                "Customer_Category"
            ]

            .value_counts()

            .reset_index()

        )


        seg_counts.columns = [

            "Category",

            "Count"

        ]


        fig10 = px.pie(

            seg_counts,

            names=
            "Category",

            values=
            "Count",

            hole=0.4,

            title=
            "Customer Category Distribution"

        )


        st.plotly_chart(
            fig10,
            use_container_width=True
        )


        seg_rev = (

            fdf.groupby(
                "Customer_Category"
            )["Total_Revenue"]

            .mean()

            .reset_index()

        )


        fig11 = px.bar(

            seg_rev,

            x=
            "Customer_Category",

            y=
            "Total_Revenue",

            color=
            "Customer_Category",

            title=
            "Average Revenue by Customer Category"

        )


        st.plotly_chart(
            fig11,
            use_container_width=True
        )


        st.subheader(
            "Explore Raw Data"
        )


        st.dataframe(
            fdf.head(200),
            use_container_width=True
        )


        st.caption(
            f"Showing first 200 of "
            f"{len(fdf):,} filtered rows."
        )

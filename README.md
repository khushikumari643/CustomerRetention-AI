# CustomerRetention AI
### An end-to-end telecom customer churn prediction, diagnosis, categorization & retention recommendation system

---

## 1. Business Understanding

**Problem statement:** A telecom provider wants to reduce customer churn by:
1. Predicting **which customers** are likely to churn and **why** (diagnostic analysis).
2. **Categorizing** customers by retention/purchase potential so retention teams can prioritize outreach.
3. **Recommending** the best-fit subscription plan to retain at-risk or upsell-ready customers.
4. Giving business stakeholders an **interactive dashboard** to monitor churn, revenue, and subscription trends.

**Success criteria:**
- Churn model ROC-AUC ≥ 0.85
- Customer categorization accuracy ≥ 0.75
- A working, deployable Streamlit web app with all 4 functional sections

**Stakeholders:** Retention/CX team, Marketing, Product/Plan design team.

---

## 2. Project Structure

```
CustomerRetentionAI/
├── app.py                              # Streamlit web application (4 sections)
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   └── customer_data_original.csv  # Original uploaded dataset (6,418 rows)
│   ├── processed/
│   │   └── customer_data_clean.csv     # Cleaned + feature-engineered dataset
│   └── subscription_plans.csv          # Custom recommendation training data (52 rows)
├── models/                             # All trained models + metadata (pickled)
│   ├── best_churn_model.pkl            # Best of 3 churn algorithms (Random Forest)
│   ├── best_category_model.pkl         # Best of 3 categorization algorithms (Random Forest)
│   ├── churn_reason_model.pkl          # Diagnostic churn-reason classifier
│   ├── recommendation_model.pkl        # Recommendation engine (1 algorithm: Random Forest)
│   ├── recommendation_price_model.pkl  # Companion price regressor (MSE/R2 evaluation)
│   ├── *_encoders.pkl / *_scaler.pkl   # Preprocessing artifacts
│   └── *_results.json                  # Metrics for every trained model
└── src/
    ├── 01_data_cleaning_fe.py          # Cleaning + feature engineering
    ├── 02_train_churn_models.py        # 3-algorithm churn model training
    ├── 03_train_category_models.py     # 3-algorithm categorization training
    ├── 04_train_recommendation_model.py# Recommendation engine training
    └── build_recommendation_dataset.py # Builds the custom subscription CSV
```

---

## 3. Data Cleaning & Feature Engineering (`src/01_data_cleaning_fe.py`)

- **Missing values** handled with business logic, not blind imputation:
  - `Value_Deal` → "No Deal" (customer isn't on a promo)
  - `Multiple_Lines` → "No" (no secondary line)
  - Internet-dependent add-ons (`Online_Security`, `Streaming_TV`, etc.) → "No" / "No Internet" (customer has no internet service, so these are structurally missing, not random)
- **Invalid values**: negative `Monthly_Charge` treated as data-entry error → replaced with median; negative charges clipped to 0.
- **New engineered features**:
  - `Tenure_Bucket` (0-6mo, 7-12mo, 13-24mo, 25-48mo, 49mo+)
  - `Total_Addon_Services` (count of Yes across 8 add-ons — engagement proxy)
  - `Avg_Monthly_Revenue`, `Extra_Charge_Ratio`, `Refund_Ratio`
  - `Is_Long_Term_Contract`, `Has_Value_Deal`, `Is_High_Spender`, `Has_Referrals`, `Is_Senior`
- **Target labels created**:
  - `Churn_Label` (binary: Churned=1 vs Stayed/Joined=0)
  - `Customer_Category` (multiclass, business-rule derived: High-Risk / Medium-Risk / Loyal / Premium-Ready) — used as the supervised target for the categorization models.

---

## 4. Modeling

### 4.1 Churn Prediction (3 algorithms)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.770 | 0.570 | 0.821 | 0.673 | 0.858 |
| **Random Forest (best)** | **0.849** | **0.757** | 0.700 | 0.728 | **0.892** |
| XGBoost | 0.824 | 0.693 | 0.697 | 0.695 | 0.889 |

Best model (Random Forest) saved as `models/best_churn_model.pkl`.

A secondary **diagnostic model** (Random Forest, trained only on churned customers) predicts the most likely `Churn_Category` (Competitor / Price / Dissatisfaction / Attitude / Other) so the app can explain *why* a customer is at risk, not just flag them.

### 4.2 Customer Categorization (3 algorithms, multiclass)
| Model | Accuracy |
|---|---|
| Logistic Regression | 0.710 |
| Decision Tree | 0.768 |
| **Random Forest (best)** | **0.812** |

Best model saved as `models/best_category_model.pkl`. Classes: `High-Risk`, `Medium-Risk`, `Loyal`, `Premium-Ready`.

### 4.3 Recommendation Engine (1 algorithm: Random Forest)
Trained on a **custom 52-row subscription-plans dataset** (`data/subscription_plans.csv`) mapping customer complaint/need profiles → recommended plan. Evaluated with 5-fold cross-validation (appropriate for this small hand-crafted catalogue):

| Metric | Value |
|---|---|
| Classification accuracy (plan) | 36.5% |
| Price regression MSE | 57.97 |
| Price regression R² | 0.814 |
| Chi-Squared (Complaint vs Plan association) | 676.0 (p < 0.001) |

The low p-value confirms `Primary_Complaint` is a statistically significant driver of the recommended plan.

---

## 5. Streamlit App (`app.py`)

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Four sections** (sidebar navigation):
1. **Churn Prediction & Diagnostic Analysis** — pick an existing customer or enter a new profile → churn probability gauge, top global risk drivers, customer-specific risk factors, and predicted churn-reason category.
2. **Customer Categorization** — predicts High-Risk / Medium-Risk / Loyal / Premium-Ready with confidence breakdown and category descriptions.
3. **Recommendation Engine** — select complaint/need profile → get the best-fit subscription plan, price estimate, and top-5 alternative matches; view full plans catalogue and model evaluation metrics (accuracy, MSE, R², Chi-Squared).
4. **Interactive Dashboard** — KPIs, churn patterns (by contract/tenure/state), revenue & tenure trends, payment/contract insights, and customer segment analysis, all built with Plotly and filterable by state/contract/status.

---

## 6. Reproducing the pipeline from scratch

```bash
cd CustomerRetentionAI
python src/01_data_cleaning_fe.py
python src/02_train_churn_models.py
python src/03_train_category_models.py
python src/build_recommendation_dataset.py
python src/04_train_recommendation_model.py
streamlit run app.py
```

## 7. Limitations & Future Work
- The dataset has no explicit signup date, so monthly/yearly trends use `Tenure_in_Months` as a cohort proxy.
- The recommendation catalogue is hand-crafted (52 rows); a production system would use real historical complaint → retention-offer outcome data.
- Diagnostic explanations combine global feature importance with per-customer business rules; a future iteration could add SHAP values for fully personalized explanations.

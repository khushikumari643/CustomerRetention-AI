"""
CustomerRetention AI - Custom Subscription Plans Dataset Builder
---------------------------------------------------------------------
Creates data/subscription_plans.csv: a hand-crafted catalogue mapping
customer complaint/need patterns + profile signals to a recommended
subscription plan. This is the training data for the recommendation
engine (one ML algorithm, as required).

Columns:
  Primary_Complaint     - main pain point category (matches Churn_Category
                           style groupings + a few extra "need" states for
                           non-churned customers)
  Contract_Preference   - Month-to-Month / One Year / Two Year
  Monthly_Charge_Band   - Low / Medium / High (customer's current spend band)
  Internet_Type_Need    - DSL / Cable / Fiber Optic / No Internet
  Family_Size_Need      - Single / Small Family / Large Family (proxy from
                           referrals/streaming needs)
  Recommended_Plan      - target label: the subscription plan to recommend
  Recommended_Price     - target numeric: monthly price of that plan (used
                           for regression-style evaluation, MSE/R2)
  Plan_Description       - explanation shown to the user in the app
"""

import pandas as pd
import os

PLAN_CATALOGUE = {
    "Value Saver Fiber":            (45.0, "Budget fiber plan, no bundled extras, cuts monthly cost ~35%."),
    "Value Saver Cable":             (35.0, "Lean cable plan for price-sensitive single users, no add-ons."),
    "Family Fiber Lite":             (55.0, "1-year discounted fiber bundle, lower effective monthly rate."),
    "Basic Connect":                 (25.0, "Entry-level DSL plan, lowest price point available."),
    "Loyalty Fiber Max":             (70.0, "Matches competitor speed/offer with loyalty discount + device upgrade."),
    "Switch-Back Saver":             (40.0, "Aggressive win-back pricing plus 3 months premium support free."),
    "Ultra Fiber Family":            (95.0, "High-speed multi-device fiber plan beating competitor offers."),
    "Premium Care Bundle":           (65.0, "Adds Premium Support + Device Protection to fix reliability issues."),
    "Reliability Plus":              (58.0, "Prioritized network reliability tier with proactive monitoring."),
    "VIP Support Plan":              (50.0, "Dedicated priority support line for service-attitude issues."),
    "White-Glove Fiber":             (75.0, "Concierge-level onboarding & dedicated support team."),
    "Standard Fiber Plus":           (48.0, "Balanced general-purpose fiber plan, moderate data allowance."),
    "Speed Boost Fiber":             (52.0, "Upgrades DSL to fiber-equivalent speed at a modest price step-up."),
    "High-Speed Cable Pro":          (68.0, "Higher-tier cable bandwidth for multi-device households."),
    "Unlimited Data Fiber":          (60.0, "Removes data caps entirely, ideal for streaming-heavy households."),
    "Unlimited Family Max":          (90.0, "Unlimited data + multi-line support for large households."),
    "Streaming Entertainment Pack":  (58.0, "Bundles TV + Movies + Music streaming add-ons at a discount."),
    "Ultimate Entertainment Bundle": (88.0, "Full streaming suite + Unlimited Data for families."),
    "SecureNet Plan":                (46.0, "Adds Online Security + Device Protection as core inclusions."),
    "SecureNet Family":              (66.0, "Household-wide online security & backup coverage."),
    "Global Talk Saver":             (42.0, "Flat-rate long-distance bundle, no per-minute overage charges."),
    "Budget Unlimited Data":         (33.0, "Low-cost plan with unlimited data to avoid extra usage fees."),
    "Premium Support Fiber":         (54.0, "24/7 premium technical support bundled with standard fiber speed."),
    "Elite Loyalty Rewards":         (85.0, "Reward plan for long-term happy customers: referral bonuses + upgrades."),
    "Loyalty Plus":                  (58.0, "2-year loyalty lock-in with modest bill credit for renewing."),
    "Flexi-Move Plan":               (50.0, "No-penalty relocation transfer with fiber availability guarantee."),
}

rows_raw = [
    ("Price", "Month-to-Month", "High", "Fiber Optic", "Single", "Value Saver Fiber"),
    ("Price", "Month-to-Month", "High", "Fiber Optic", "Small Family", "Value Saver Fiber"),
    ("Price", "Month-to-Month", "Medium", "Cable", "Single", "Value Saver Cable"),
    ("Price", "Month-to-Month", "Medium", "Cable", "Small Family", "Value Saver Cable"),
    ("Price", "One Year", "High", "Fiber Optic", "Small Family", "Family Fiber Lite"),
    ("Price", "One Year", "High", "Fiber Optic", "Large Family", "Family Fiber Lite"),
    ("Price", "Month-to-Month", "Low", "DSL", "Single", "Basic Connect"),
    ("Price", "Month-to-Month", "Low", "DSL", "Small Family", "Basic Connect"),
    ("Competitor", "Month-to-Month", "High", "Fiber Optic", "Small Family", "Loyalty Fiber Max"),
    ("Competitor", "Month-to-Month", "High", "Fiber Optic", "Large Family", "Loyalty Fiber Max"),
    ("Competitor", "Month-to-Month", "Medium", "Cable", "Single", "Switch-Back Saver"),
    ("Competitor", "Month-to-Month", "Medium", "Cable", "Small Family", "Switch-Back Saver"),
    ("Competitor", "One Year", "High", "Fiber Optic", "Large Family", "Ultra Fiber Family"),
    ("Competitor", "Two Year", "High", "Fiber Optic", "Large Family", "Ultra Fiber Family"),
    ("Dissatisfaction", "Month-to-Month", "Medium", "Fiber Optic", "Small Family", "Premium Care Bundle"),
    ("Dissatisfaction", "One Year", "Medium", "Fiber Optic", "Small Family", "Premium Care Bundle"),
    ("Dissatisfaction", "Month-to-Month", "High", "Cable", "Single", "Reliability Plus"),
    ("Dissatisfaction", "Month-to-Month", "High", "Cable", "Small Family", "Reliability Plus"),
    ("Attitude", "Month-to-Month", "Medium", "Cable", "Single", "VIP Support Plan"),
    ("Attitude", "Month-to-Month", "Medium", "DSL", "Single", "VIP Support Plan"),
    ("Attitude", "Month-to-Month", "High", "Fiber Optic", "Small Family", "White-Glove Fiber"),
    ("Attitude", "One Year", "High", "Fiber Optic", "Small Family", "White-Glove Fiber"),
    ("Other", "One Year", "Medium", "Fiber Optic", "Single", "Standard Fiber Plus"),
    ("Other", "One Year", "Medium", "Fiber Optic", "Small Family", "Standard Fiber Plus"),
    ("Slow Speed", "Month-to-Month", "Medium", "DSL", "Single", "Speed Boost Fiber"),
    ("Slow Speed", "Month-to-Month", "Medium", "DSL", "Small Family", "Speed Boost Fiber"),
    ("Slow Speed", "One Year", "High", "Cable", "Small Family", "High-Speed Cable Pro"),
    ("Slow Speed", "One Year", "High", "Cable", "Large Family", "High-Speed Cable Pro"),
    ("Data Limits", "Month-to-Month", "Medium", "Fiber Optic", "Small Family", "Unlimited Data Fiber"),
    ("Data Limits", "One Year", "Medium", "Fiber Optic", "Small Family", "Unlimited Data Fiber"),
    ("Data Limits", "Two Year", "High", "Fiber Optic", "Large Family", "Unlimited Family Max"),
    ("Data Limits", "Two Year", "High", "Fiber Optic", "Small Family", "Unlimited Family Max"),
    ("Streaming Needs", "Month-to-Month", "Medium", "Fiber Optic", "Small Family", "Streaming Entertainment Pack"),
    ("Streaming Needs", "One Year", "Medium", "Fiber Optic", "Small Family", "Streaming Entertainment Pack"),
    ("Streaming Needs", "One Year", "High", "Fiber Optic", "Large Family", "Ultimate Entertainment Bundle"),
    ("Streaming Needs", "Two Year", "High", "Fiber Optic", "Large Family", "Ultimate Entertainment Bundle"),
    ("Security Concerns", "Month-to-Month", "Medium", "Fiber Optic", "Single", "SecureNet Plan"),
    ("Security Concerns", "One Year", "Medium", "Fiber Optic", "Single", "SecureNet Plan"),
    ("Security Concerns", "One Year", "High", "Fiber Optic", "Small Family", "SecureNet Family"),
    ("Security Concerns", "Two Year", "High", "Fiber Optic", "Small Family", "SecureNet Family"),
    ("Long Distance Charges", "Month-to-Month", "Medium", "Cable", "Small Family", "Global Talk Saver"),
    ("Long Distance Charges", "One Year", "Medium", "Cable", "Small Family", "Global Talk Saver"),
    ("Extra Data Charges", "Month-to-Month", "Low", "DSL", "Single", "Budget Unlimited Data"),
    ("Extra Data Charges", "Month-to-Month", "Low", "DSL", "Small Family", "Budget Unlimited Data"),
    ("Support Quality", "One Year", "Medium", "Fiber Optic", "Single", "Premium Support Fiber"),
    ("Support Quality", "Two Year", "Medium", "Fiber Optic", "Single", "Premium Support Fiber"),
    ("None / Satisfied", "Two Year", "High", "Fiber Optic", "Large Family", "Elite Loyalty Rewards"),
    ("None / Satisfied", "Two Year", "High", "Fiber Optic", "Small Family", "Elite Loyalty Rewards"),
    ("None / Satisfied", "Two Year", "Medium", "Fiber Optic", "Small Family", "Loyalty Plus"),
    ("None / Satisfied", "One Year", "Medium", "Fiber Optic", "Single", "Loyalty Plus"),
    ("Moved / Relocation", "Month-to-Month", "Medium", "Fiber Optic", "Small Family", "Flexi-Move Plan"),
    ("Moved / Relocation", "Month-to-Month", "Medium", "Fiber Optic", "Single", "Flexi-Move Plan"),
]

rows = []
for complaint, contract, band, itype, fam, plan in rows_raw:
    price, desc = PLAN_CATALOGUE[plan]
    rows.append((complaint, contract, band, itype, fam, plan, price, desc))

df = pd.DataFrame(rows, columns=[
    "Primary_Complaint", "Contract_Preference", "Monthly_Charge_Band",
    "Internet_Type_Need", "Family_Size_Need", "Recommended_Plan",
    "Recommended_Price", "Plan_Description"
])

out_dir = "/home/claude/CustomerRetentionAI/data"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/subscription_plans.csv"
df.to_csv(out_path, index=False)
print(f"Saved {df.shape[0]} rows -> {out_path}")
print(df["Recommended_Plan"].nunique(), "unique plans")

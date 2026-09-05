import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Historical CRM Opportunity Records
training_records = {
    'company_size': [15, 250, 1200, 45, 500, 30, 2000, 85, 1500, 10],
    'industry': ['Tech', 'Finance', 'Healthcare', 'Tech', 'Retail', 'Tech', 'Finance', 'Manufacturing', 'Tech', 'Retail'],
    'site_visits_last_30d': [2, 14, 25, 5, 18, 1, 30, 8, 22, 3],
    'whitepaper_downloads': [0, 2, 4, 1, 2, 0, 5, 1, 3, 0],
    'email_click_rate': [0.05, 0.40, 0.65, 0.15, 0.50, 0.0, 0.70, 0.20, 0.55, 0.10],
    'days_in_pipeline': [60, 22, 14, 45, 18, 90, 12, 35, 15, 75],
    'converted': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]  # Target: Won (1) or Lost (0)
}

train_df = pd.DataFrame(training_records)
features = ['company_size', 'industry', 'site_visits_last_30d', 'whitepaper_downloads', 'email_click_rate', 'days_in_pipeline']
X_train = train_df[features]
y_train = train_df['converted']

# 2. Build Preprocessing & Modeling Pipeline
numeric_features = ['company_size', 'site_visits_last_30d', 'whitepaper_downloads', 'email_click_rate', 'days_in_pipeline']
categorical_features = ['industry']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

lead_scorer = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42, min_samples_leaf=2))
])

lead_scorer.fit(X_train, y_train)

# 3. Score Inbound Leads & Assign Operational Tiers
new_leads_data = {
    'lead_id': ['LEAD-901', 'LEAD-902', 'LEAD-903'],
    'lead_name': ['Acro Corp', 'BioGenix Labs', 'Solo Studio'],
    'company_size': [800, 35, 8],
    'industry': ['Finance', 'Healthcare', 'Tech'],
    'site_visits_last_30d': [19, 6, 2],
    'whitepaper_downloads': [3, 1, 0],
    'email_click_rate': [0.58, 0.25, 0.05],
    'days_in_pipeline': [9, 30, 42]
}
new_leads_df = pd.DataFrame(new_leads_data)

# Predict probability of closing
probs = lead_scorer.predict_proba(new_leads_df[features])[:, 1]
new_leads_df['win_probability_pct'] = (probs * 100).round(1)

def assign_tier(prob):
    if prob >= 70.0:
        return 'HOT (Assign AE within 1 hour)'
    elif prob >= 40.0:
        return 'WARM (Nurture sequence)'
    return 'COLD (Automated email only)'

new_leads_df['sales_action'] = new_leads_df['win_probability_pct'].apply(assign_tier)

print("=== INBOUND LEAD CONVERSION SCORES ===")
output_cols = ['lead_id', 'lead_name', 'win_probability_pct', 'sales_action']
print(new_leads_df[output_cols].to_string(index=False))
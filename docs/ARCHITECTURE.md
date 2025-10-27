# 📊 Project Architecture & Flow Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────┐         ┌─────────────────────────────────┐  │
│  │   Streamlit Dashboard    │         │      FastAPI REST API           │  │
│  │   (Port 8501)           │         │      (Port 8000)                │  │
│  │                          │         │                                 │  │
│  │  🏠 Home                 │         │  GET  /health                   │  │
│  │  🔮 Prediction           │         │  GET  /model/info               │  │
│  │  🔍 Explanation          │         │  POST /predict                  │  │
│  │  👥 Segmentation         │         │  POST /predict/batch            │  │
│  │  📈 Performance          │         │  POST /predict/file             │  │
│  └──────────────────────────┘         │  POST /explain                  │  │
│                                       │  GET  /segments/info            │  │
│                                       └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Load Models
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL LAYER                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │  Churn Pipeline   │  │  Segmentation    │  │   SHAP Explainer       │  │
│  │  (pipeline.pkl)   │  │  (K-Means)       │  │   (explainer.pkl)      │  │
│  │                   │  │  (segment.pkl)   │  │                        │  │
│  │  • RF             │  │  • 4 Clusters    │  │  • TreeExplainer       │  │
│  │  • XGBoost        │  │  • PCA Viz       │  │  • Force Plots         │  │
│  │  • LightGBM ⭐    │  │  • Profiles      │  │  • Waterfall Plots     │  │
│  │  • Ensemble       │  │  • Insights      │  │  • Feature Importance  │  │
│  └───────────────────┘  └──────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Train/Process
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE PROCESSING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: Custom Transformers                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  TenureGroupTransformer  →  Groups tenure into bins                  │  │
│  │  FeatureEngineer         →  Creates new intelligent features         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Step 2: Column Transformer                                                 │
│  ┌─────────────────────┐           ┌─────────────────────────────────┐    │
│  │  Numerical Pipeline │           │  Categorical Pipeline            │    │
│  │  ├─ Imputer         │           │  ├─ Imputer                      │    │
│  │  └─ RobustScaler    │           │  └─ OneHotEncoder                │    │
│  └─────────────────────┘           └─────────────────────────────────┘    │
│                                                                              │
│  Step 3: Imbalance Handling                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  SMOTEENN / ADASYN  →  Balance 73-27 class distribution             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Step 4: Classification                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  LightGBM Classifier  →  Predict [No Churn | Churn]                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Process
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Telco_customer_churn.csv  (7,043 customers × 20 features)           │ │
│  ├───────────────────────────────────────────────────────────────────────┤ │
│  │  Demographics: gender, SeniorCitizen, Partner, Dependents            │ │
│  │  Account:      tenure, Contract, PaperlessBilling, PaymentMethod     │ │
│  │  Services:     PhoneService, InternetService, OnlineSecurity, ...    │ │
│  │  Financial:    MonthlyCharges, TotalCharges                          │ │
│  │  Target:       Churn (Yes/No)                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. DATA INGESTION                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │  load_and_clean()            │
                    │  (preprocess.py)             │
                    │                              │
                    │  • Load CSV                  │
                    │  • Handle missing values     │
                    │  • Create tenure groups      │
                    │  • Label encode              │
                    │  • Remove customerID         │
                    └──────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          2. MODEL TRAINING                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │  ML Pipeline Training    │      │  Segmentation Training   │
    │  (pipeline.py)           │      │  (segmentation.py)       │
    │                          │      │                          │
    │  1. Split data (80/20)   │      │  1. Prepare features     │
    │  2. Build pipeline       │      │  2. Find optimal k       │
    │  3. Train 5 models       │      │  3. Fit K-Means          │
    │     • RF                 │      │  4. Create profiles      │
    │     • XGBoost            │      │  5. Generate insights    │
    │     • LightGBM ⭐        │      │  6. Visualize            │
    │     • Voting             │      └──────────────────────────┘
    │     • Stacking           │                  │
    │  4. Cross-validate       │                  │
    │  5. Select best          │                  │
    │  6. Save model           │                  │
    └──────────────────────────┘                  │
                    │                              │
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │  SHAP Explainer Setup        │
                    │  (explainability.py)         │
                    │                              │
                    │  1. Create TreeExplainer     │
                    │  2. Calculate SHAP values    │
                    │  3. Generate plots           │
                    │  4. Feature importance       │
                    │  5. Save explainer           │
                    └──────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          3. MODEL ARTIFACTS                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼────────────────────────┐
            │                       │                        │
            ▼                       ▼                        ▼
    ┌──────────────┐      ┌──────────────────┐     ┌─────────────────┐
    │ Pipeline.pkl │      │ Segmentation.pkl │     │ Explainer.pkl   │
    │              │      │                  │     │                 │
    │ 81.2% Acc    │      │ 4 Segments       │     │ SHAP Values     │
    └──────────────┘      └──────────────────┘     └─────────────────┘
            │                       │                        │
            └───────────────────────┼────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          4. DEPLOYMENT                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │  Streamlit App           │      │  FastAPI Server          │
    │  (streamlit_app.py)      │      │  (api.py)                │
    │                          │      │                          │
    │  • Load models           │      │  • Load models           │
    │  • Create UI             │      │  • Define endpoints      │
    │  • Handle inputs         │      │  • Validate data         │
    │  • Display results       │      │  • Return JSON           │
    │  • Show visualizations   │      │  • OpenAPI docs          │
    └──────────────────────────┘      └──────────────────────────┘
                    │                              │
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │  Docker Compose              │
                    │  (docker-compose.yml)        │
                    │                              │
                    │  • Package both services     │
                    │  • Define networking         │
                    │  • Mount volumes             │
                    │  • Health checks             │
                    └──────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          5. PRODUCTION                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │  Users Access            │      │  Systems Integrate       │
    │                          │      │                          │
    │  • Web browser           │      │  • CRM systems           │
    │  • Input customer data   │      │  • Batch processing      │
    │  • Get predictions       │      │  • Automated alerts      │
    │  • View explanations     │      │  • Data pipelines        │
    │  • Explore segments      │      │  • BI dashboards         │
    └──────────────────────────┘      └──────────────────────────┘
```

---

## Feature Engineering Flow

```
Raw Customer Data
        │
        ├─► gender (M/F)
        ├─► SeniorCitizen (0/1)
        ├─► tenure (months)
        ├─► MonthlyCharges ($)
        ├─► TotalCharges ($)
        ├─► Contract (type)
        └─► [15 more features...]
        │
        ▼
┌───────────────────────┐
│ TenureGroupTransformer│
└───────────────────────┘
        │
        ├─► Create: tenure_group
        │   • 1-12 months
        │   • 13-24 months
        │   • 25-36 months
        │   • etc.
        │
        ▼
┌───────────────────────┐
│   FeatureEngineer     │
└───────────────────────┘
        │
        ├─► Create: AvgChargesPerMonth
        │   = TotalCharges / tenure
        │
        ├─► Create: ServiceUsageScore
        │   = Count of services used
        │
        ├─► Create: ChargeToServiceRatio
        │   = MonthlyCharges / ServiceUsageScore
        │
        └─► [More engineered features...]
        │
        ▼
┌───────────────────────┐
│  ColumnTransformer    │
└───────────────────────┘
        │
        ├─► Numerical Features
        │   ├─ Impute (median)
        │   └─ Scale (RobustScaler)
        │
        └─► Categorical Features
            ├─ Impute (most_frequent)
            └─ Encode (OneHotEncoder)
        │
        ▼
┌───────────────────────┐
│   SMOTEENN/ADASYN     │
└───────────────────────┘
        │
        └─► Balanced Dataset
            • Churn: ~50%
            • No Churn: ~50%
        │
        ▼
┌───────────────────────┐
│   LightGBM Model      │
└───────────────────────┘
        │
        └─► Prediction: [0.22, 0.78]
                         │     │
                         │     └─► Churn Prob: 78%
                         └─────► No Churn Prob: 22%
```

---

## SHAP Explanation Flow

```
Customer Input + Trained Model
        │
        ▼
┌───────────────────────┐
│  SHAP TreeExplainer   │
└───────────────────────┘
        │
        ├─► For each feature:
        │   "What if this feature changed?"
        │
        ▼
Feature Contributions:
├─► Contract (Month-to-month):  +0.15  ⚠️ INCREASES churn risk
├─► Tenure (6 months):           +0.12  ⚠️ INCREASES churn risk
├─► OnlineSecurity (No):         +0.08  ⚠️ INCREASES churn risk
├─► MonthlyCharges ($85):        +0.06  ⚠️ INCREASES churn risk
├─► Partner (Yes):               -0.02  ✅ DECREASES churn risk
└─► TotalCharges ($510):         -0.04  ✅ DECREASES churn risk
        │
        ▼
┌───────────────────────┐
│  Visualization        │
└───────────────────────┘
        │
        ├─► Force Plot
        │   [Shows push/pull effects]
        │
        ├─► Waterfall Plot
        │   [Step-by-step breakdown]
        │
        ├─► Summary Plot
        │   [Global importance]
        │
        └─► Dependence Plot
            [Feature interactions]
```

---

## Customer Segmentation Flow

```
All Customer Data (7,043 customers)
        │
        ▼
┌─────────────────────────┐
│  Extract Features       │
│  (Numerical only)       │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  StandardScaler         │
│  (Normalize features)   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  K-Means Clustering     │
│  (k=4 clusters)         │
└─────────────────────────┘
        │
        ├────────────┬────────────┬────────────┐
        │            │            │            │
        ▼            ▼            ▼            ▼
    Segment 0    Segment 1    Segment 2    Segment 3
    ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
    │ 1,200  │   │ 1,800  │   │ 2,000  │   │ 2,043  │
    │ custs  │   │ custs  │   │ custs  │   │ custs  │
    ├────────┤   ├────────┤   ├────────┤   ├────────┤
    │ 45%    │   │ 15%    │   │ 28%    │   │ 18%    │
    │ churn  │   │ churn  │   │ churn  │   │ churn  │
    ├────────┤   ├────────┤   ├────────┤   ├────────┤
    │ High   │   │ Loyal  │   │ At-    │   │ Budget │
    │ Risk   │   │ Long-  │   │ Risk   │   │ Stable │
    │ New    │   │ Term   │   │ Mid    │   │ Basic  │
    └────────┘   └────────┘   └────────┘   └────────┘
        │            │            │            │
        └────────────┴────────────┴────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Business Insights            │
        │                               │
        │  • Segment characteristics    │
        │  • Churn patterns             │
        │  • Retention strategies       │
        │  • Targeted campaigns         │
        └───────────────────────────────┘
```

---

## Deployment Architecture

```
                    ┌─────────────────────────────────┐
                    │         Internet                 │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
        ┌──────────────────────┐      ┌──────────────────────┐
        │   Web Browser        │      │   HTTP Client        │
        │   (User)             │      │   (Integration)      │
        └──────────────────────┘      └──────────────────────┘
                    │                                │
                    │                                │
                    ▼                                ▼
        ┌──────────────────────┐      ┌──────────────────────┐
        │   :8501              │      │   :8000              │
        │   Streamlit          │      │   FastAPI            │
        │   Container          │◄─────┤   Container          │
        └──────────────────────┘      └──────────────────────┘
                    │                                │
                    └────────────┬───────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Shared Volumes       │
                    │                        │
                    │  • models/             │
                    │    ├─ pipeline.pkl     │
                    │    ├─ segment.pkl      │
                    │    └─ explainer.pkl    │
                    │                        │
                    │  • data/               │
                    │    └─ Telco_*.csv      │
                    └────────────────────────┘
```

This comprehensive breakdown should give you a complete understanding of how everything works together! 🚀

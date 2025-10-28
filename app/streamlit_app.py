"""
Streamlit Web Application for Customer Churn Prediction
Multi-page interactive dashboard with prediction, explanation, and segmentation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import ChurnPredictionPipeline
from segmentation import CustomerSegmentation
from explainability import ModelExplainer
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        # Load prediction pipeline
        pipeline = ChurnPredictionPipeline.load_model('models/churn_model_pipeline.pkl')
        
        # Load segmentation model
        segmenter = CustomerSegmentation.load_model('models/segmentation_model.pkl')
        
        # Load explainer
        try:
            explainer_data = joblib.load('models/shap_explainer.pkl')
            explainer = ModelExplainer(pipeline.pipeline)
            explainer.explainer = explainer_data['explainer']
            explainer.shap_values = explainer_data.get('shap_values')
            explainer.feature_names = explainer_data.get('feature_names')
        except:
            explainer = ModelExplainer(pipeline.pipeline)
            explainer.create_explainer(method='tree')
        
        return pipeline, segmenter, explainer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please train the models first by running the training scripts.")
        return None, None, None


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features to match training data encoding.
    Uses LabelEncoder-compatible mappings (alphabetical order).
    """
    df = df.copy()
    
    # Binary encodings (alphabetical: Female=0, Male=1; No=0, Yes=1)
    binary_mappings = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1}
    }
    
    # MultipleLines encoding (alphabetical)
    multiple_lines_map = {'No': 0, 'No phone service': 1, 'Yes': 2}
    
    # Internet service addons (alphabetical: No=0, No internet service=1, Yes=2)
    addon_map = {'No': 0, 'No internet service': 1, 'Yes': 2}
    
    # InternetService (alphabetical: DSL=0, Fiber optic=1, No=2)
    internet_service_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    
    # Contract (alphabetical: Month-to-month=0, One year=1, Two year=2)
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    
    # PaymentMethod (alphabetical)
    payment_method_map = {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1,
        'Electronic check': 2,
        'Mailed check': 3
    }
    
    # Apply binary mappings
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # Apply complex mappings
    if 'MultipleLines' in df.columns:
        df['MultipleLines'] = df['MultipleLines'].map(multiple_lines_map)
    
    if 'InternetService' in df.columns:
        df['InternetService'] = df['InternetService'].map(internet_service_map)
    
    if 'Contract' in df.columns:
        df['Contract'] = df['Contract'].map(contract_map)
    
    if 'PaymentMethod' in df.columns:
        df['PaymentMethod'] = df['PaymentMethod'].map(payment_method_map)
    
    # Apply addon mappings
    addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                  'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in addon_cols:
        if col in df.columns:
            df[col] = df[col].map(addon_map)
    
    return df


def create_input_form():
    """Create input form for customer data"""
    st.markdown('<p class="sub-header">📝 Enter Customer Information</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    with col2:
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", 
                                      "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, 5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 
                                       float(tenure * monthly_charges), 50.0)
    
    with col3:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    # Create dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    return input_data


def prediction_page():
    """Main prediction page"""
    st.markdown('<p class="main-header">🎯 Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    pipeline, segmenter, explainer = load_models()
    
    if pipeline is None:
        return
    
    # Input form
    input_data = create_input_form()
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("Analyzing customer data..."):
            try:
                # Encode categorical features
                input_data_encoded = encode_categorical_features(input_data)
                
                # Make prediction
                prediction = pipeline.predict(input_data_encoded)[0]
                prediction_proba = pipeline.predict_proba(input_data_encoded)[0]
                
                # Display results
                st.markdown('<p class="sub-header">📊 Prediction Results</p>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Prediction",
                        value="⚠️ CHURN" if prediction == 1 else "✅ NO CHURN",
                        delta="High Risk" if prediction == 1 else "Low Risk"
                    )
                
                with col2:
                    st.metric(
                        label="Churn Probability",
                        value=f"{prediction_proba[1]*100:.1f}%",
                        delta=f"{prediction_proba[1]*100 - 50:.1f}% from baseline"
                    )
                
                with col3:
                    st.metric(
                        label="Retention Probability",
                        value=f"{prediction_proba[0]*100:.1f}%",
                        delta=f"{50 - prediction_proba[0]*100:.1f}% from baseline"
                    )
                
                # Risk gauge
                st.markdown('<p class="sub-header">🎚️ Risk Assessment</p>', unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Risk Score", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#90EE90'},
                            {'range': [30, 70], 'color': '#FFD700'},
                            {'range': [70, 100], 'color': '#FF6347'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown('<p class="sub-header">💡 Recommended Actions</p>', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.error("⚠️ **HIGH CHURN RISK DETECTED**")
                    st.markdown("""
                    **Immediate Actions Required:**
                    - 🎁 Offer retention incentives (e.g., discount, service upgrade)
                    - 📞 Schedule proactive customer outreach call
                    - 🔍 Review account for service issues or complaints
                    - 💼 Consider personalized retention offer
                    - 📧 Send targeted retention campaign
                    """)
                else:
                    st.success("✅ **LOW CHURN RISK**")
                    st.markdown("""
                    **Suggested Actions:**
                    - 🌟 Maintain excellent service quality
                    - 📈 Explore upselling opportunities
                    - 🎯 Include in loyalty program
                    - 💬 Request feedback/testimonial
                    - 🔄 Monitor satisfaction regularly
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


def explanation_page():
    """Model explanation page with SHAP"""
    st.markdown('<p class="main-header">🔍 Prediction Explanation</p>', unsafe_allow_html=True)
    st.markdown("Understand which factors contribute most to churn predictions")
    st.markdown("---")
    
    pipeline, segmenter, explainer = load_models()
    
    if pipeline is None:
        return
    
    # Input form
    input_data = create_input_form()
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        explain_button = st.button("🔬 Explain Prediction", use_container_width=True, type="primary")
    
    if explain_button:
        with st.spinner("Generating explanation..."):
            try:
                # Encode categorical features
                input_data_encoded = encode_categorical_features(input_data)
                
                # Get prediction
                prediction = pipeline.predict(input_data_encoded)[0]
                prediction_proba = pipeline.predict_proba(input_data_encoded)[0]
                
                # Get SHAP explanation
                # Prepare data
                from preprocess import load_and_clean
                df = load_and_clean('data/Telco_customer_churn.csv')
                X = df.drop('Churn', axis=1)
                
                # Concatenate for explanation (use encoded data)
                X_explain = pd.concat([X.iloc[:1], input_data_encoded], ignore_index=True)
                
                explanation = explainer.explain_instance(X_explain, index=1)
                
                # Display prediction
                st.markdown('<p class="sub-header">📊 Prediction Summary</p>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", "CHURN ⚠️" if prediction == 1 else "NO CHURN ✅")
                with col2:
                    st.metric("Churn Probability", f"{prediction_proba[1]*100:.1f}%")
                
                # Feature contributions
                st.markdown('<p class="sub-header">🎯 Top Feature Contributions</p>', unsafe_allow_html=True)
                
                contributions = explanation['feature_contributions']
                top_features = list(contributions.items())[:10]
                
                # Create bar chart
                features = [f[0] for f in top_features]
                values = [f[1] for f in top_features]
                colors = ['red' if v > 0 else 'green' for v in values]
                
                fig = go.Figure(go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{v:.4f}" for v in values],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="SHAP Feature Importance (Top 10)",
                    xaxis_title="SHAP Value (Impact on Churn Probability)",
                    yaxis_title="Feature",
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.markdown('<p class="sub-header">📖 Interpretation Guide</p>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **🔴 Red bars (Positive values):**
                    - Increase churn probability
                    - Risk factors for customer leaving
                    - Focus areas for intervention
                    """)
                
                with col2:
                    st.markdown("""
                    **🟢 Green bars (Negative values):**
                    - Decrease churn probability
                    - Retention factors
                    - Positive customer attributes
                    """)
                
                # Detailed breakdown
                st.markdown('<p class="sub-header">📋 Detailed Feature Analysis</p>', unsafe_allow_html=True)
                
                analysis_df = pd.DataFrame([
                    {
                        'Feature': feature,
                        'Value': explanation['instance_values'][feature],
                        'SHAP Value': contribution,
                        'Impact': 'Increases Churn Risk' if contribution > 0 else 'Decreases Churn Risk',
                        'Strength': abs(contribution)
                    }
                    for feature, contribution in top_features
                ])
                
                st.dataframe(
                    analysis_df.style.background_gradient(subset=['SHAP Value'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error generating explanation: {str(e)}")
                st.exception(e)


def segmentation_page():
    """Customer segmentation dashboard"""
    st.markdown('<p class="main-header">👥 Customer Segmentation Analysis</p>', unsafe_allow_html=True)
    st.markdown("Explore distinct customer groups and their characteristics")
    st.markdown("---")
    
    pipeline, segmenter, explainer = load_models()
    
    if segmenter is None:
        return
    
    # Load data
    from preprocess import load_and_clean
    df = load_and_clean('data/Telco_customer_churn.csv')
    
    # Get segments
    df['Segment'] = segmenter.predict(df)
    
    # Overview
    st.markdown('<p class="sub-header">📊 Segment Overview</p>', unsafe_allow_html=True)
    
    if segmenter.segment_profiles is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Number of Segments", segmenter.n_clusters)
        with col3:
            st.metric("Average Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
        with col4:
            st.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
        
        # Segment sizes
        st.markdown('<p class="sub-header">📈 Segment Distribution</p>', unsafe_allow_html=True)
        
        segment_counts = df['Segment'].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Segment {i}" for i in segment_counts.index],
                y=segment_counts.values,
                text=segment_counts.values,
                textposition='auto',
                marker=dict(color=px.colors.qualitative.Set3[:len(segment_counts)])
            )
        ])
        
        fig.update_layout(
            title="Customer Count by Segment",
            xaxis_title="Segment",
            yaxis_title="Number of Customers",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment profiles
        st.markdown('<p class="sub-header">📋 Segment Profiles</p>', unsafe_allow_html=True)
        
        profiles_display = segmenter.segment_profiles.copy()
        
        # Safe formatting function to handle None/NaN values
        def safe_format_pct(x):
            return f"{x:.1f}%" if pd.notna(x) else "N/A"
        
        def safe_format_currency(x):
            return f"${x:.2f}" if pd.notna(x) else "N/A"
        
        def safe_format_months(x):
            return f"{x:.1f} months" if pd.notna(x) else "N/A"
        
        profiles_display['Churn_Rate'] = profiles_display['Churn_Rate'].apply(safe_format_pct)
        profiles_display['Size_Pct'] = profiles_display['Size_Pct'].apply(safe_format_pct)
        profiles_display['Avg_MonthlyCharges'] = profiles_display['Avg_MonthlyCharges'].apply(safe_format_currency)
        profiles_display['Avg_TotalCharges'] = profiles_display['Avg_TotalCharges'].apply(safe_format_currency)
        profiles_display['Avg_Tenure'] = profiles_display['Avg_Tenure'].apply(safe_format_months)
        
        st.dataframe(profiles_display, use_container_width=True)
        
        # Churn rate comparison
        st.markdown('<p class="sub-header">⚠️ Churn Risk by Segment</p>', unsafe_allow_html=True)
        
        churn_by_segment = df.groupby('Segment')['Churn'].mean() * 100
        
        colors = ['red' if x > 30 else 'orange' if x > 20 else 'green' 
                 for x in churn_by_segment.values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Segment {i}" for i in churn_by_segment.index],
                y=churn_by_segment.values,
                text=[f"{v:.1f}%" for v in churn_by_segment.values],
                textposition='auto',
                marker=dict(color=colors)
            )
        ])
        
        fig.add_hline(
            y=df['Churn'].mean() * 100,
            line_dash="dash",
            line_color="black",
            annotation_text="Overall Average"
        )
        
        fig.update_layout(
            title="Churn Rate by Segment",
            xaxis_title="Segment",
            yaxis_title="Churn Rate (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment insights
        st.markdown('<p class="sub-header">💡 Segment Insights</p>', unsafe_allow_html=True)
        
        insights = segmenter.generate_segment_insights(df)
        
        for segment_id, insight in insights.items():
            with st.expander(f"📌 Segment {segment_id} Details"):
                st.text(insight)


def model_performance_page():
    """Model performance metrics page"""
    st.markdown('<p class="main-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown("Comprehensive evaluation of model performance")
    st.markdown("---")
    
    pipeline, segmenter, explainer = load_models()
    
    if pipeline is None:
        return
    
    st.markdown('<p class="sub-header">🎯 Model Information</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Model Type:** {pipeline.model_type.upper()}
        
        **Resampling Method:** {pipeline.resampling_method.upper() if pipeline.resampling_method else 'None'}
        
        **Number of Features:** {len(pipeline.feature_names) if pipeline.feature_names else 'N/A'}
        """)
    
    with col2:
        st.success("""
        **Key Features:**
        - ✅ Production-ready sklearn Pipeline
        - ✅ Automated preprocessing
        - ✅ Class imbalance handling
        - ✅ Ensemble methods support
        """)
    
    st.markdown('<p class="sub-header">📊 Performance Metrics</p>', unsafe_allow_html=True)
    
    # Display actual model performance metrics (updated after retraining)
    metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
    
    with metrics_col1:
        st.metric("Accuracy", "74.0%", "Improved")
    with metrics_col2:
        st.metric("Precision", "50.7%", "Balanced")
    with metrics_col3:
        st.metric("Recall", "77.3%", "High")
    with metrics_col4:
        st.metric("F1-Score", "61.2%", "Improved")
    with metrics_col5:
        st.metric("ROC-AUC", "82.2%", "Excellent")
    
    st.markdown("---")
    
    st.info("""
    💡 **Note:** For detailed model training results and performance comparisons, 
    please refer to the training notebooks and logs in the `models/` directory.
    """)


def main():
    """Main application"""
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "🔮 Prediction", "🔍 Explanation", "👥 Segmentation", "📈 Model Performance"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    
    This application provides:
    - Real-time churn prediction
    - SHAP-based explanations
    - Customer segmentation
    - Model performance metrics
    
    ### Technologies Used
    - Python
    - Scikit-learn
    - LightGBM
    - SHAP
    - Streamlit
    - Plotly
    
    ### Team
    Customer Churn Prediction Project
    """)
    
    # Route to pages
    if page == "🏠 Home":
        st.markdown('<p class="main-header">🎯 Customer Churn Prediction System</p>', unsafe_allow_html=True)
        st.markdown("### Welcome to the Advanced Customer Churn Prediction Platform")
        st.markdown("---")
        
        st.markdown("""
        ## 🚀 Overview
        
        This comprehensive machine learning system helps businesses:
        - **Predict** customer churn with high accuracy
        - **Explain** why customers are likely to leave
        - **Segment** customers into distinct groups
        - **Take action** with data-driven recommendations
        
        ## 🎯 Key Features
        
        ### 1. 🔮 Real-Time Prediction
        - Input customer data and get instant churn predictions
        - Risk assessment with probability scores
        - Actionable recommendations
        
        ### 2. 🔍 Model Explainability (SHAP)
        - Understand which features drive predictions
        - See the impact of each customer attribute
        - Transparent, interpretable AI
        
        ### 3. 👥 Customer Segmentation
        - K-Means clustering identifies customer groups
        - Analyze churn patterns by segment
        - Targeted retention strategies
        
        ### 4. 📈 Performance Monitoring
        - Track model metrics
        - Validation results
        - Production-ready pipeline
        
        ## 🛠️ Technology Stack
        
        - **Machine Learning:** Scikit-learn, LightGBM, XGBoost
        - **Explainability:** SHAP
        - **Imbalance Handling:** SMOTEENN, ADASYN
        - **Visualization:** Plotly, Matplotlib, Seaborn
        - **Deployment:** Streamlit, FastAPI, Docker
        
        ## 📊 Getting Started
        
        Use the sidebar to navigate through different sections:
        
        1. **Prediction**: Make churn predictions for individual customers
        2. **Explanation**: Understand what drives each prediction
        3. **Segmentation**: Explore customer segments and patterns
        4. **Performance**: View model metrics and validation results
        
        ---
        
        **👉 Select a page from the sidebar to begin!**
        """)
        
    elif page == "🔮 Prediction":
        prediction_page()
    
    elif page == "🔍 Explanation":
        explanation_page()
    
    elif page == "👥 Segmentation":
        segmentation_page()
    
    elif page == "📈 Model Performance":
        model_performance_page()


if __name__ == "__main__":
    main()

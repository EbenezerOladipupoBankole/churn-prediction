import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
import shap
import warnings
import numpy as np
from streamlit_option_menu import option_menu
import os
from src.clv_analysis import analyze_clv

st.set_page_config(page_title="Customer Insights Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for modern UI ---
st.markdown("""
<style>
    /* General App Styling */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 2rem;
    }

    /* Sidebar Styling */
    .st-emotion-cache-163ttbj {
        background-color: #1E1E2D;
    }
    .st-emotion-cache-6qob1r {
        background-color: #1E1E2D;
    }

    /* Card-like containers */
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-1v0mbdj {
        background-color: #1E1E2D;
        border: 1px solid #2c2c44;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .st-emotion-cache-1r4qj8v:hover, .st-emotion-cache-1v0mbdj:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }

    /* Metric Cards */
    .st-emotion-cache-1fcdl2p {
        background: linear-gradient(135deg, #6e45e2 0%, #88d3ce 100%);
        border-radius: 10px;
        padding: 25px;
        color: white;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.3);
    }
    .st-emotion-cache-1fcdl2p .st-emotion-cache-1g8m226, .st-emotion-cache-1fcdl2p .st-emotion-cache-1wiv0i5 {
        color: white;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #6e45e2 0%, #88d3ce 100%);
        color: white;
        border: none;
        padding: 12px 0px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        opacity: 0.8;
        border: none;
        color: white;
    }

</style>
""", unsafe_allow_html=True)

# Load models & preprocessors (cached)
@st.cache_resource
def load_models():
    logistic = joblib.load("models/logistic.pkl")
    rf = joblib.load("models/rf.pkl")
    xgb = joblib.load("models/xgb.pkl")
    return {"logistic": logistic, "rf": rf, "xgb": xgb}

@st.cache_data
def load_test_data():
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop(columns=["Churn", "customerID", "tenure_bucket", "expected_tenure", "CLV"])
    y_test = test_df["Churn"].apply(lambda x: 1 if x == 'Yes' else 0)
    return X_test, y_test

@st.cache_data
def load_full_processed_data():
    """Loads and concatenates all processed data splits."""
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)

# SHAP Helper Functions (moved to global scope)
@st.cache_resource
def get_shap_explainer(_model_pipeline, data):
    """Create and cache the SHAP explainer."""
    model = _model_pipeline.named_steps['classifier']
    preprocessor = _model_pipeline.named_steps['preprocessor']
    data_transformed = preprocessor.transform(data)
    return shap.TreeExplainer(model, data_transformed)

@st.cache_data
def get_shap_values(_explainer, _model_pipeline, data):
    """Calculate and cache SHAP values for a given dataset."""
    preprocessor = _model_pipeline.named_steps['preprocessor']
    data_transformed = preprocessor.transform(data)
    return _explainer(data_transformed)

# --- App Layout ---
models = load_models()
X_test, y_test = load_test_data()

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("## ChurnGuard AI") # Placeholder for a logo
    st.markdown("---")
    # Introduction Section
    profile_pic_path = "assets/eb.jpg"
    if os.path.exists(profile_pic_path):
        st.image(profile_pic_path, width=150)
    else:
        st.info("Add `assets/eb.jpg` to see your profile picture here.")
    st.markdown("Bankole Ebenezer")
    st.markdown("*Data Scientist | ML Engineer*")
    st.markdown("Welcome to my churn prediction dashboard! Feel free to connect with me via the links below.")
    st.markdown("[LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)")
    st.markdown("---")
    page = option_menu(
        menu_title=None,
        options=["🔮 Predict Churn", "📊 Model Performance", "💰 CLV Overview"],
        icons=["person-bounding-box", "bar-chart-line-fill", "cash-coin"],
        menu_icon="cast",
        default_index=0,
    )

# --- Page 1: Predict Churn ---
if page == "🔮 Predict Churn":
    st.title("🔮 Customer Churn Prediction")
    st.write("Enter customer details into the form below to generate a real-time churn prediction and explanation.")

    form_container = st.container(border=True)
    with form_container:
      with st.form("prediction_form"):
        st.subheader("Customer Details")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Customer Profile")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has a partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])

        with col2:
            st.subheader("Contract & Billing")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=1.0)

        with st.expander("🔧 Additional Services (Advanced)"):
            # Create columns for a cleaner layout inside the expander
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="multi")
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
            with exp_col2:
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="security")
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup")
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device")
            with exp_col3:
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech")
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="tv")
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="movies")

        submitted = st.form_submit_button("🔮 Predict Churn")

    if submitted:
        # Create a dataframe from the inputs
        input_data = {
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
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
            'PaperlessBilling': [paperless],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        input_df = pd.DataFrame(input_data)

        with st.spinner("Analyzing customer data..."):
            # --- Replicate Feature Engineering from data_prep.py ---
            service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            
            # Count 'Yes' values for services
            input_df['services_count'] = input_df[service_cols].apply(lambda row: sum(row == "Yes"), axis=1)
            
            # Create flag for internet but no tech support
            input_df['internet_no_support'] = ((input_df['InternetService'] != 'No') & (input_df['TechSupport'] == 'No')).astype(int)
            
            # Create ratio feature, handling potential division by zero
            denominator = (input_df['tenure'] * input_df['MonthlyCharges']).replace(0, 1)
            input_df['monthly_to_total_ratio'] = input_df['TotalCharges'] / denominator

            # --- Make prediction ---
            xgb_pipeline = models['xgb']
            churn_proba = xgb_pipeline.predict_proba(input_df)[:, 1][0]
            churn_percentage = churn_proba * 100

        # --- Display results ---
        st.subheader("Prediction Result")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(
                label="Churn Probability",
                value=f"{churn_percentage:.2f}%",
                delta="High Risk" if churn_percentage > 50 else "Medium Risk" if churn_percentage > 25 else "Low Risk",
                delta_color="inverse"
            )
            st.progress(float(churn_proba))
            st.write("This bar represents the model's predicted probability of churn.")

        with res_col2:
            with st.container(border=True):
                st.subheader("Prediction Explanation (SHAP)")
                st.write("The plot below shows which customer attributes are pushing the churn risk up (red) or down (blue) for this specific prediction.")
                # Explain the model's prediction using SHAP
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    explainer = get_shap_explainer(models['xgb'], X_test)
                    shap_values = get_shap_values(explainer, models['xgb'], input_df)

                    st.write(f"**Model Base Value (Average Churn Probability):** {explainer.expected_value:.2f}")
                    fig, ax = plt.subplots(facecolor='#1E1E2D')
                    ax.tick_params(colors='white')
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig, use_container_width=True)

# --- Page 2: Model Performance ---
elif page == "📊 Model Performance":
    st.title("📊 Overall Model Performance")
    st.write("Comparing Precision, Recall, F1-Score, and AUC for all trained models on the test dataset.")

    with st.spinner("Calculating model metrics..."):
        metrics_data = []
        roc_data = {}
        for name, model_pipeline in models.items():
            y_proba = model_pipeline.predict_proba(X_test)[:, 1]
            y_pred = model_pipeline.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            metrics_data.append({
                "Model": name.upper(),
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "AUC": roc_auc
            })
            roc_data[name] = (fpr, tpr, roc_auc)

        metrics_df = pd.DataFrame(metrics_data).set_index("Model")

    with st.container(border=True):
        st.subheader("Performance Metrics")
        st.dataframe(metrics_df.style.format("{:.3f}").background_gradient(cmap='viridis'), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(facecolor='#1E1E2D')
        for name, (fpr, tpr, roc_auc) in roc_data.items():
            ax.plot(fpr, tpr, label=f'{name.upper()} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'w--', label='Chance (AUC = 0.500)')
        ax.set_xlabel('False Positive Rate', color='white')
        ax.set_ylabel('True Positive Rate', color='white')
        ax.set_title('Receiver Operating Characteristic', color='white')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        st.pyplot(fig)

    with col2, st.container(border=True):
        st.subheader("Global Feature Importance (XGBoost)")
        st.write("Average impact of each feature on the model's predictions across the entire test set.")
        with st.spinner("Calculating SHAP values... This may take a moment."):
            explainer = get_shap_explainer(models['xgb'], X_test)
            preprocessor = models['xgb'].named_steps['preprocessor']
            X_test_transformed = preprocessor.transform(X_test)
            feature_names = preprocessor.get_feature_names_out()
            shap_values = explainer(X_test_transformed)
            fig_shap, ax_shap = plt.subplots(facecolor='#1E1E2D')
            ax_shap.tick_params(colors='white')
            shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, plot_type="bar", show=False, plot_size=None)
            plt.title("Global Feature Importance", color='white')
            st.pyplot(fig_shap)

# --- Page 3: CLV Overview ---
elif page == "💰 CLV Overview":
    st.title("💰 Customer Lifetime Value (CLV) Overview")
    st.write("This section analyzes the entire customer base to understand the relationship between Customer Lifetime Value (CLV) and churn behavior.")

    with st.spinner("Analyzing CLV data..."):
        full_df = load_full_processed_data()
        df_analyzed, churn_by_quartile = analyze_clv(full_df)

    col1, col2 = st.columns(2)

    with col1, st.container(border=True):
        st.subheader("CLV Distribution")
        fig, ax = plt.subplots(facecolor='#1E1E2D')
        ax.hist(df_analyzed['CLV'], bins=30, color='#88d3ce')
        ax.set_title("Distribution of Customer Lifetime Value", color='white')
        ax.set_xlabel("CLV ($)", color='white')
        ax.set_ylabel("Number of Customers", color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    with col2, st.container(border=True):
        st.subheader("Churn Rate by CLV Quartile")
        churn_by_quartile['churn_rate_pct'] = churn_by_quartile['churn_rate'] * 100
        fig, ax = plt.subplots(facecolor='#1E1E2D')
        ax.bar(churn_by_quartile['clv_quartile'], churn_by_quartile['churn_rate_pct'], color='#6e45e2')
        ax.set_title("Churn Rate Increases with CLV", color='white')
        ax.set_xlabel("CLV Quartile", color='white')
        ax.set_ylabel("Churn Rate (%)", color='white')
        ax.tick_params(colors='white')
        ax.grid(axis='y', alpha=0.2)
        st.pyplot(fig)

    with st.container(border=True):
        st.subheader("Actionable Recommendation")
        st.markdown("""
        **Insight:** The analysis reveals a counter-intuitive trend: customers in the **High** and **Premium** CLV quartiles have a significantly higher churn rate than those in the lower quartiles. This suggests that our most valuable customers are also the most at-risk.

        **Recommendation:** Prioritize retention efforts on the **High** and **Premium** CLV segments. These customers, often on month-to-month contracts with high monthly charges (e.g., for fiber optic internet), are valuable but lack long-term commitment. Proactive outreach with loyalty discounts, contract upgrade incentives, or dedicated support could be highly effective in reducing revenue loss from this critical group.
        """)

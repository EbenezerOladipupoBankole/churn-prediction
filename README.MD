# Customer Churn Prediction & Customer Lifetime Value (CLV)

## 📌 Business Context
SaaS companies lose 5–7% of revenue annually to churn.  
This project predicts churn and estimates Customer Lifetime Value (CLV) so retention efforts can prioritize the most valuable customers.

## 🔧 Features
- Predict churn probability with 3 models:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
- Customer Lifetime Value (CLV) analysis & quartiles
- SHAP/feature importance explanations
- Deployed Streamlit app

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- A virtual environment tool (like `venv`)

### 1. Setup the Environment

First, clone the repository and navigate into the project directory. Then, create a virtual environment and install the required packages.

```bash
# Create and activate a virtual environment
python -m venv venv 
```


# Activate the virtual environment
# On Windows:
`venv\Scripts\activate`
# On macOS/Linux:
`source venv/bin/activate`

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data & Train Models

You need to run the data preparation script first, followed by the model training script.

```bash
# This will download the raw data and create processed splits
python src/data_prep.py

# This will train the models and save them to the /models directory
python train_models.py
```

> **Note:** The `train_models.py` script is essential for creating the model files (`logistic.pkl`, `rf.pkl`, etc.) that the Streamlit app depends on.

### 3. Run the Streamlit App

Once the models are trained and saved, you can launch the web application.

```bash
streamlit run app.py
```

## � CLV Assumptions
- **Expected Tenure**:
  - Month-to-month: 12 months
  - One-year contract: 24 months
  - Two-year contract: 36 months
- **CLV Formula**:  

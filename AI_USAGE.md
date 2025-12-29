# AI Usage Documentation

This document outlines how AI assistance (Gemini Code Assist) was used in the development of the Customer Churn Prediction project.

## 1. Key Areas of AI Assistance

I used the AI assistant primarily for the following tasks:


*   **Debugging:** Identifying why models were failing (due to unencoded categorical data) and suggesting the `ColumnTransformer` approach.
*   **Conceptual Understanding:** Explaining the difference between `st.cache_data` and `st.cache_resource` and how to correctly implement SHAP explanations for different model types.
*   **Best Practices:** Suggesting improvements like saving the entire model pipeline instead of just the model, and ensuring the Streamlit app loads models and data efficiently.

## 2. Example Prompts That Mattered

Here are a few prompts that led to significant progress:



1.  **Solving a Core Problem:**
    > "My `train_models.py` script is failing because the models can't handle string values. How can I create a scikit-learn pipeline to properly preprocess my categorical and numerical features and save it so I can use it in my Streamlit app?"

2.  **Implementing Interpretability:**
    > "Show me how to implement a SHAP summary plot for my XGBoost model in the 'Model Performance' tab of my Streamlit app. Make sure it's calculated efficiently."

## 3. Verification and Corrections

While the AI provided excellent starting points, I had to perform the following checks and corrections:

*   **Feature Selection:** The initial AI-generated preprocessing pipeline included features that should have been dropped (e.g., `customerID`, `CLV`). I manually adjusted the feature list to prevent data leakage and improve model relevance.
*   **Pathing and Naming:** I verified all file paths and model names to ensure consistency between the training script and the Streamlit app.
*   **App Logic:** I refined the logic in the Streamlit prediction tab to handle user inputs gracefully and present the output (churn probability, CLV) in a clear, user-friendly format. The AI provided the function, but I customized the presentation.

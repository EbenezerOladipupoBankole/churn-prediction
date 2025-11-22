# c:\Users\HomePC\churn-prediction\src\clv_analysis.py
import pandas as pd

"""
This script will contain the functions for CLV analysis, including quartile segmentation and churn rate calculations.
"""

def analyze_clv(df):
    """
    Analyzes CLV by segmenting customers into quartiles and calculating churn rates.

    Args:
        df (pd.DataFrame): DataFrame containing 'CLV' and 'Churn' columns.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The original DataFrame with an added 'clv_quartile' column.
            - pd.DataFrame: A summary DataFrame with churn rates per CLV quartile.
    """
    # Create CLV quartiles
    df['clv_quartile'] = pd.qcut(df['CLV'], 4, labels=["Low", "Medium", "High", "Premium"])

    # Calculate churn rate by quartile
    churn_rate_by_quartile = df.groupby('clv_quartile', observed=True)['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
    churn_rate_by_quartile.rename(columns={'Churn': 'churn_rate'}, inplace=True)

    return df, churn_rate_by_quartile
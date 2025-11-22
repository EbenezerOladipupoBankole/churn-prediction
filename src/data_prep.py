import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import requests

RAW_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_PATH = "data/processed/"

def load_data():
    # Ensure the raw data directory exists
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

    # Download the data if it doesn't exist locally
    if not os.path.exists(RAW_PATH):
        print(f"Downloading data from {RAW_URL}...")
        response = requests.get(RAW_URL, timeout=30) # Set a 30-second timeout
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(RAW_PATH, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("✅ Download complete.")

    df = pd.read_csv(RAW_PATH)
    return df

def preprocess(df):
    # Handle TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Feature engineering
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, df["tenure"].max()],
        labels=["0-6m", "6-12m", "12-24m", "24m+"]
    )
    df["services_count"] = df[
        ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
         "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    ].apply(lambda row: sum(row == "Yes"), axis=1)

    df["internet_no_support"] = (
        (df["InternetService"] != "No") &
        (df["TechSupport"] == "No")
    ).astype(int)

    df["monthly_to_total_ratio"] = df["TotalCharges"] / (df["tenure"] * df["MonthlyCharges"]).replace(0, 1)

    # CLV assumption
    contract_map = {"Month-to-month": 12, "One year": 24, "Two year": 36}
    df["expected_tenure"] = df["Contract"].map(contract_map)
    df["CLV"] = df["MonthlyCharges"] * df["expected_tenure"]

    return df

def split_and_save(df):
    train, temp = train_test_split(df, test_size=0.4, stratify=df["Churn"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["Churn"], random_state=42)

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    train.to_csv(PROCESSED_PATH + "train.csv", index=False)
    val.to_csv(PROCESSED_PATH + "val.csv", index=False)
    test.to_csv(PROCESSED_PATH + "test.csv", index=False)

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    split_and_save(df)
    print("✅ Data preprocessed & saved in data/processed/")

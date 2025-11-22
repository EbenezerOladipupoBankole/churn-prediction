import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def load_processed():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    return train, val, test

def train_models(train, val):
    # Define features and target
    # Drop engineered features not directly used by the model or that leak target info
    features_to_drop = ["Churn", "customerID", "tenure_bucket", "expected_tenure", "CLV"]
    X_train = train.drop(columns=features_to_drop)
    y_train = train["Churn"].apply(lambda x: 1 if x == 'Yes' else 0)
    X_val = val.drop(columns=features_to_drop)
    y_val = val["Churn"].apply(lambda x: 1 if x == 'Yes' else 0)

    # Identify categorical and numerical features
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_features = X_train.select_dtypes(include=['int64', 'float64', 'int32']).columns

    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
        "xgb": XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=200, random_state=42)
    }
    
    results = {}
    os.makedirs("models", exist_ok=True)

    # Fit and save the preprocessor
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    for name, model in models.items():
        # Create a full pipeline with preprocessor and model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        
        # Train the model
        pipeline.fit(X_train, y_train)

        # Evaluate on validation set
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        results[name] = auc

        # Save the entire pipeline
        joblib.dump(pipeline, f"models/{name}.pkl")

    return results

if __name__ == "__main__":
    train, val, test = load_processed()
    results = train_models(train, val)
    print("✅ Models and preprocessor trained and saved.")
    print("Validation AUC scores:")
    for name, score in results.items():
        print(f"  - {name}: {score:.4f}")

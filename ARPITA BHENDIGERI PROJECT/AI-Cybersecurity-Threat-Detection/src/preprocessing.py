import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data():
    data = pd.read_csv("data/custom_dataset.csv")
    print("✅ Data Loaded")
    return data

def clean_data(data):
    print("✅ Data Cleaned")
    return data

def split_data(data):
    X = data.drop("Attack_Label", axis=1)
    y = data["Attack_Label"]
    print("✅ Data Split into X and y")
    return X, y

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Data Scaled")
    return X_scaled

def split_train_test(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
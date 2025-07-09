# --- src/data_prep.py ---
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df.loc[:, col] = le.fit_transform(df[col])

    return df
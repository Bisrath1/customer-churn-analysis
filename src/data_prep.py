import pandas as pd
from sklearn.preprocessing import LabelEncoder


df =  pd.read_csv(r'C:\10x AIMastery\customer-churn-analysis\data\Telco_Customer_Churn_Dataset.csv')
print(df.head())


def clean_data(df):
    """Cleans and preprocesses the data."""
    # Drop customerID (not useful for modeling)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert 'TotalCharges' to numeric (may contain spaces or bad data)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Handle missing values
    df = df.dropna()

    # Encode binary Yes/No columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Encode remaining categorical columns using Label Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df

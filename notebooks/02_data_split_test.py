import sys
import os
import pandas as pd

# Add ../src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_prep import  clean_data
from data_split import split_data

df = pd.read_csv('data/Telco_Customer_Churn_Dataset.csv')
df_cleaned = clean_data(df)

X_train, X_test, y_train, y_test = split_data(df_cleaned)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

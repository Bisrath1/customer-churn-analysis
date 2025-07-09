# --- notebooks/02_data_split_test.py ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_prep import load_data, clean_data
from data_split import split_data

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco_Customer_Churn_Dataset.csv'))
df = load_data(file_path)
df_clean = clean_data(df)

X_train, X_test, y_train, y_test = split_data(df_clean)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train target distribution:\n", y_train.value_counts(normalize=True))
print("Test target distribution:\n", y_test.value_counts(normalize=True))

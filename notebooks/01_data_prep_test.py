# --- notebooks/01_data_prep_test.py ---
import sys, os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_prep import load_data, clean_data

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco_Customer_Churn_Dataset.csv'))
df = load_data(file_path)
print("Raw Data Sample:\n", df.head())

df_clean = clean_data(df)
print("\nCleaned Data Sample:\n", df_clean.head())
print("\nCleaned Data Info:\n")
print(df_clean.info())




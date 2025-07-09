# --- notebooks/03_feature_select_test.py ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_prep import load_data, clean_data
from data_split import split_data
from feature_select import select_features

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco_Customer_Churn_Dataset.csv'))
df = load_data(file_path)
df_clean = clean_data(df)
X_train, X_test, y_train, y_test = split_data(df_clean)

top_features = select_features(X_train, y_train, k=10)
print("Top Selected Features:")
print(top_features)







# --- notebooks/04_train_model_test.py ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_prep import load_data, clean_data
from data_split import split_data
from feature_select import select_features
from train_model import train_model, evaluate_model

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco_Customer_Churn_Dataset.csv'))
df = load_data(file_path)
df_clean = clean_data(df)
X_train, X_test, y_train, y_test = split_data(df_clean)

top_features = select_features(X_train, y_train, k=10)
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]


model = train_model(X_train_sel, y_train, model_type='logistic')
report = evaluate_model(model, X_test_sel, y_test)
print("📊 Logistic Regression Report:\n", report)

model_rf = train_model(X_train_sel, y_train, model_type='rf')
report_rf = evaluate_model(model_rf, X_test_sel, y_test)
print("🌲 Random Forest Report:\n", report_rf)


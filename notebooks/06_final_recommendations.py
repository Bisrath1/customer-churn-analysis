# --- notebooks/05_final_recommendations.py ---
import sys, os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_prep import load_data, clean_data
from data_split import split_data
from feature_select import select_features
from train_model import train_model, evaluate_model

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco_Customer_Churn_Dataset.csv'))
df = load_data(file_path)
df_clean = clean_data(df)
X_train, X_test, y_train, y_test = split_data(df_clean)

# Select top 10 features
selected_features = select_features(X_train, y_train, k=10)
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# Train final model
final_model = train_model(X_train_sel, y_train, model_type='rf')
report, cm, (fpr, tpr, roc_auc) = evaluate_model(final_model, X_test_sel, y_test)

# --- Task 6: Final Recommendations ---
recommendations = f"""
🔍 **Final Insights & Recommendations**

✅ **Model Summary:**
- Final model: Random Forest Classifier
- Accuracy: {report['accuracy']:.2f}
- Precision (Churn): {report['1']['precision']:.2f}
- Recall (Churn): {report['1']['recall']:.2f}
- AUC Score: {roc_auc:.2f}

📌 **Key Features Impacting Churn:**
{', '.join(selected_features)}

💡 **Recommendations:**
1. Focus on customers with short tenure, high monthly charges, and month-to-month contracts.
2. Consider offering incentives or loyalty programs to high-risk customers.
3. Enhance support and engagement strategies to reduce voluntary churn.

📈 **Next Steps:**
- Deploy the trained model for real-time churn prediction.
- Monitor performance and retrain periodically with updated data.
- Integrate churn risk scores into the customer dashboard for sales teams.
"""

summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'final_recommendations.txt'))
with open(summary_path, 'w') as f:
    f.write(recommendations)

print(recommendations)

import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_prep import load_data, clean_data
from data_split import split_data
from feature_select import select_features

# --- Load + Clean ---
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco_Customer_Churn_Dataset.csv'))
df = load_data(file_path)
df_clean = clean_data(df)

# --- Split + Feature Select ---
X_train, X_test, y_train, y_test = split_data(df_clean)
top_features = select_features(X_train, y_train, k=10)
X_train_sel, X_test_sel = X_train[top_features], X_test[top_features]

# --- Train Model for Evaluation Plots ---
model = RandomForestClassifier(random_state=42)
model.fit(X_train_sel, y_train)
y_pred = model.predict(X_test_sel)
y_prob = model.predict_proba(X_test_sel)[:, 1]

# --- Metrics ---
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)


# === 📊 Customer Churn Distribution ===
plt.figure(figsize=(8, 6))
sns.countplot(data=df_clean, x='Churn',
              palette=['#FF6B6B', '#4ECDC4'],
              edgecolor='black', linewidth=1.5)

plt.title("Customer Churn Distribution", fontsize=16, weight='bold', pad=20, color='#333333')
plt.xlabel("Churn Status", fontsize=14, color='#333333')
plt.ylabel("Count", fontsize=14, color='#333333')
plt.xticks(fontsize=12, color='#333333')
plt.yticks(fontsize=12, color='#333333')

# Add count labels
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='#333333', xytext=(0, 5),
                textcoords='offset points')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#666666')
ax.spines['bottom'].set_color('#666666')
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Save
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'churn_distribution_enhanced.png'))
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# === 🔥 Feature Correlation Heatmap ===
plt.figure(figsize=(10, 8))
corr_matrix = df_clean.select_dtypes(include=[np.number]).corr()

sns.heatmap(corr_matrix,
            annot=True,
            cmap='RdYlBu',
            center=0,
            vmin=-1, vmax=1,
            fmt='.2f',
            annot_kws={'size': 10, 'color': '#333333'},
            square=True,
            linewidths=0.5,
            linecolor='white')

plt.title("Feature Correlation Heatmap", fontsize=16, weight='bold', pad=20, color='#333333')
plt.xticks(rotation=45, ha='right', fontsize=12, color='#333333')
plt.yticks(fontsize=12, color='#333333')

# Remove outer spines
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

# Save
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'correlation_heatmap_enhanced.png'))
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()


# === Confusion Matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.abspath(os.path.join(r'C:\10x AIMastery\customer-churn-analysis\outputs\confusion_matrix.png')))
plt.show()

# === ROC Curve ===
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.abspath(os.path.join(r'C:\10x AIMastery\customer-churn-analysis\outputs\roc_curve.png')))
plt.show()

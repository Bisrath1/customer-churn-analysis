# --- src/train_model.py ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)
# --- src/evaluate_model.py ---
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a classification model on the test data.
    
    Returns:
    - classification report as dict
    - confusion matrix
    - fpr, tpr, auc for ROC curve
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities for ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For models like SVM without predict_proba
        y_proba = model.decision_function(X_test)

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return report, cm, (fpr, tpr, roc_auc)

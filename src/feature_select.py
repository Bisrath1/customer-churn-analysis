# --- src/feature_select.py ---
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_features(X, y, k=10):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)
    return list(X.columns[selector.get_support()])
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_features(X, y, k=10):
    """
    Selects the top k features based on mutual information.

    Parameters:
    - X: Feature dataframe
    - y: Target labels
    - k: Number of top features to select

    Returns:
    - selected_features: List of top feature names
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)

    selected_columns = X.columns[selector.get_support()]
    return list(selected_columns)

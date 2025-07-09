import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_column='Churn', test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Parameters:
    - df (DataFrame): Cleaned dataset
    - target_column (str): Target column name
    - test_size (float): Proportion of test data
    - random_state (int): Seed for reproducibility

    Returns:
    - X_train, X_test, y_train, y_test
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

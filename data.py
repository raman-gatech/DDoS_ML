
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CFG

def load_ddos_data():
    cfg = CFG()
    df = pd.read_csv(cfg.dataset_path)

    # Drop ID-like columns if present
    for col in cfg.id_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    if cfg.label_col not in df.columns:
        raise ValueError(f"Label column '{cfg.label_col}' not found in dataset")

    y = df[cfg.label_col]
    X = df.drop(columns=[cfg.label_col])

    # Handle basic missing values: fill numeric with median, categorical with mode
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0])

    # One-hot encode categoricals
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y
    )

    return X_train, X_val, y_train, y_val

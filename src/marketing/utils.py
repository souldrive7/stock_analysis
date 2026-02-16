from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ID_CANDIDATES = ["row_id", "customer_id", "id", "client_id", "contact_id"]
POSITIVE_LABELS = {"yes", "y", "1", "true", "t", "positive"}
TARGET_CANDIDATES = ["y", "target", "label", "response"]


def detect_target_column(df: pd.DataFrame, explicit_target: Optional[str] = None) -> str:
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(
                f"Target column '{explicit_target}' not found. Available columns: {list(df.columns)}"
            )
        return explicit_target

    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "Target column could not be auto-detected. "
        f"Available columns: {list(df.columns)}"
    )


def encode_binary_target(y_raw: pd.Series) -> Tuple[np.ndarray, str]:
    if y_raw.dtype == object:
        y = y_raw.astype(str).str.strip().str.lower()
        uniq = sorted(y.dropna().unique().tolist())
        if len(uniq) != 2:
            raise ValueError(f"Target must be binary. Found labels: {uniq}")
        positive = next((u for u in uniq if u in POSITIVE_LABELS), uniq[-1])
        return (y == positive).astype(int).to_numpy(), positive

    uniq = sorted(pd.Series(y_raw).dropna().unique().tolist())
    if len(uniq) != 2:
        raise ValueError(f"Target must be binary. Found values: {uniq}")
    positive = uniq[-1]
    return (pd.Series(y_raw) == positive).astype(int).to_numpy(), str(positive)


def detect_identifier_column(df: pd.DataFrame) -> Optional[str]:
    for col in ID_CANDIDATES:
        if col in df.columns:
            return col
    return None


def fit_fold_preprocessor(X_train: pd.DataFrame) -> Dict[str, object]:
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    num_fill = {col: X_train[col].median() for col in num_cols}
    cat_fill = {}
    for col in cat_cols:
        mode = X_train[col].mode(dropna=True)
        cat_fill[col] = mode.iloc[0] if not mode.empty else "unknown"

    X_num = X_train[num_cols].copy()
    for col in num_cols:
        X_num[col] = X_num[col].fillna(num_fill[col])

    if cat_cols:
        X_cat = X_train[cat_cols].copy()
        for col in cat_cols:
            X_cat[col] = X_cat[col].fillna(cat_fill[col]).astype(str)
        X_cat = pd.get_dummies(X_cat, columns=cat_cols, dummy_na=False)
        X_proc = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X_proc = X_num.reset_index(drop=True)

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_fill": num_fill,
        "cat_fill": cat_fill,
        "feature_columns": X_proc.columns.tolist(),
    }


def transform_fold_features(X: pd.DataFrame, prep: Dict[str, object]) -> pd.DataFrame:
    num_cols: List[str] = prep["num_cols"]
    cat_cols: List[str] = prep["cat_cols"]
    feature_columns: List[str] = prep["feature_columns"]
    num_fill: Dict[str, float] = prep["num_fill"]
    cat_fill: Dict[str, str] = prep["cat_fill"]

    X_num = X[num_cols].copy() if num_cols else pd.DataFrame(index=X.index)
    for col in num_cols:
        X_num[col] = X_num[col].fillna(num_fill[col])

    if cat_cols:
        X_cat = X[cat_cols].copy()
        for col in cat_cols:
            X_cat[col] = X_cat[col].fillna(cat_fill[col]).astype(str)
        X_cat = pd.get_dummies(X_cat, columns=cat_cols, dummy_na=False)
        X_proc = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X_proc = X_num.reset_index(drop=True)

    return X_proc.reindex(columns=feature_columns, fill_value=0.0)

from typing import Dict, List

import numpy as np
import pandas as pd


def build_call_score(
    oof_df: pd.DataFrame,
    schema: Dict[str, object],
    score_mode: str = "mean_proba",
) -> pd.Series:
    model_columns = schema["model_columns"]
    required_models: List[str] = schema["required_models"]
    optional_models: List[str] = schema["optional_models"]

    components = []
    for model in required_models:
        proba_col = model_columns[model]["proba"]
        if proba_col not in oof_df.columns:
            raise ValueError(f"Missing required probability column: {proba_col}")
        components.append(oof_df[proba_col])

    for model in optional_models:
        proba_col = model_columns[model]["proba"]
        unc_col = model_columns[model]["uncertainty"]
        has_optional_proba = proba_col in oof_df.columns and oof_df[proba_col].notna().any()
        has_edl_pair = (
            proba_col in oof_df.columns
            and unc_col in oof_df.columns
            and oof_df[proba_col].notna().any()
            and oof_df[unc_col].notna().any()
        )
        if model == "edl" and score_mode == "edl_uncertainty_weighted":
            if not has_edl_pair:
                raise ValueError(
                    "score_mode='edl_uncertainty_weighted' requires both "
                    f"non-NaN '{proba_col}' and '{unc_col}'. "
                    "Run with --enable-edl or switch --score-mode mean_proba."
                )
            unc = pd.to_numeric(oof_df[unc_col], errors="coerce").clip(0.0, 1.0)
            components.append(pd.to_numeric(oof_df[proba_col], errors="coerce") * (1.0 - unc))
            continue

        if has_optional_proba:
            components.append(oof_df[proba_col])

    if not components:
        raise ValueError(
            "No prediction columns available to build score. "
            "Check model outputs in oof_results.csv."
        )

    return pd.concat(components, axis=1).mean(axis=1)


def perfect_gain_curve(y_true: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    y = (y > 0).astype(float)

    n = len(y)
    total_true = float(y.sum())
    if n == 0 or total_true <= 0.0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    positive_rate = total_true / float(n)
    if positive_rate >= 1.0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    return np.array([0.0, positive_rate, 1.0]), np.array([0.0, 1.0, 1.0])

from typing import Dict, List

import pandas as pd

PROBA_PREFIX = "y_pred_proba_"
UNC_PREFIX = "y_pred_uncertainty_"
REQUIRED_MODELS = ["lgbm", "logit", "nn"]
OPTIONAL_MODELS = ["edl"]


def list_pred_columns(oof_df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in oof_df.columns if c.startswith(prefix)]


def build_score_schema() -> Dict[str, object]:
    model_columns = {}
    for model in REQUIRED_MODELS + OPTIONAL_MODELS:
        model_columns[model] = {
            "proba": f"{PROBA_PREFIX}{model}",
            "uncertainty": f"{UNC_PREFIX}{model}",
        }

    return {
        "required_models": REQUIRED_MODELS,
        "optional_models": OPTIONAL_MODELS,
        "model_columns": model_columns,
    }

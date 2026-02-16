import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

from src.config import Config
from src.marketing.scoring import perfect_gain_curve
from src.marketing.utils import fit_fold_preprocessor, transform_fold_features
from src.models.lgbm import LGBMModel


def _pred_cols(oof_df: pd.DataFrame) -> List[str]:
    return [c for c in oof_df.columns if c.startswith("y_pred_proba_")]


def plot_marketing_gain_chart(
    oof_df: pd.DataFrame,
    output_dir: str,
    include_score_curve: bool = True,
) -> Optional[str]:
    if "y_true" not in oof_df.columns:
        return None
    total_true = float(oof_df["y_true"].sum())
    if total_true <= 0:
        print("[Marketing][Gain] skipped: no positive samples in y_true.")
        return None

    plt.figure(figsize=(10, 7))
    for col in _pred_cols(oof_df):
        sorted_df = oof_df.sort_values(col, ascending=False).reset_index(drop=True)
        gain = sorted_df["y_true"].cumsum() / total_true
        pct = (np.arange(len(sorted_df)) + 1) / len(sorted_df)
        plt.plot(
            pct,
            gain,
            label="Marketing " + col.replace("y_pred_proba_", "proba_"),
        )

    if include_score_curve and "score" in oof_df.columns:
        sorted_df = oof_df.sort_values("score", ascending=False).reset_index(drop=True)
        gain = sorted_df["y_true"].cumsum() / total_true
        pct = (np.arange(len(sorted_df)) + 1) / len(sorted_df)
        plt.plot(
            pct,
            gain,
            label="Marketing score_rank",
            linewidth=2.5,
            color="black",
        )

    perfect_x, perfect_y = perfect_gain_curve(oof_df["y_true"])
    plt.plot(
        perfect_x,
        perfect_y,
        linestyle="--",
        color="green",
        linewidth=2.0,
        label="Marketing perfect",
    )
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Marketing random")
    plt.title("Marketing Gain Chart")
    plt.xlabel("Fraction Contacted")
    plt.ylabel("Cumulative Gain")
    plt.grid(True, linestyle=":")
    plt.legend(title="Marketing Models")
    plt.tight_layout()
    out = os.path.join(output_dir, "marketing_gain_chart.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_marketing_calibration_curve(oof_df: pd.DataFrame, output_dir: str) -> Optional[str]:
    if "y_true" not in oof_df.columns or oof_df["y_true"].sum() == 0:
        print("[Marketing][Calibration] skipped: no positive samples in y_true.")
        return None

    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Marketing perfect")
    for col in _pred_cols(oof_df):
        try:
            prob_true, prob_pred = calibration_curve(
                oof_df["y_true"], oof_df[col], n_bins=20, strategy="uniform"
            )
            plt.plot(
                prob_pred,
                prob_true,
                marker="o",
                label="Marketing " + col.replace("y_pred_proba_", "proba_"),
            )
        except Exception as e:
            print(f"[Marketing][Calibration] skipped column '{col}': {e}")

    plt.title("Marketing Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.grid(True, linestyle=":")
    plt.legend(title="Marketing Models")
    plt.tight_layout()
    out = os.path.join(output_dir, "marketing_calibration_curve.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_marketing_shap_lgbm(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    output_dir: str,
    random_state: int = 42,
) -> Optional[str]:
    try:
        import shap
    except Exception as e:
        print(
            "[Marketing][SHAP] skipped: failed to import shap. "
            f"Reason: {e}. Next action: run `pip install -r requirements.txt`."
        )
        return None

    try:
        prep = fit_fold_preprocessor(X_raw)
        X_proc = transform_fold_features(X_raw, prep)
        X_train, X_val, y_train, y_val = train_test_split(
            X_proc, y, test_size=0.2, random_state=random_state, stratify=y
        )

        model = LGBMModel(dict(Config.LGBM_PARAMS))
        model.train(X_train, y_train, X_val, y_val)

        sample_n = min(5000, len(X_proc))
        sample = X_proc.sample(sample_n, random_state=random_state)
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        plt.title("Marketing SHAP Summary (LGBM)", y=1.02)
        plt.tight_layout()
        out = os.path.join(output_dir, "marketing_shap_summary_lgbm.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return out
    except Exception as e:
        print(
            "[Marketing][SHAP] skipped due to runtime error. "
            f"Reason: {e}. Next action: check LightGBM/SHAP compatibility and retry."
        )
        return None


def plot_marketing_uncertainty_hist_edl(oof_df: pd.DataFrame, output_dir: str) -> Optional[str]:
    col = "y_pred_uncertainty_edl"
    if col not in oof_df.columns:
        return None
    plt.figure(figsize=(10, 6))
    plt.hist(oof_df[col], bins=30, edgecolor="black", alpha=0.8)
    plt.title("Marketing EDL Uncertainty Histogram")
    plt.xlabel("uncertainty_edl")
    plt.ylabel("count")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    out = os.path.join(output_dir, "marketing_uncertainty_hist_edl.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_marketing_uncertainty_vs_score(oof_df: pd.DataFrame, output_dir: str) -> Optional[str]:
    u_col = "y_pred_uncertainty_edl"
    if u_col not in oof_df.columns or "score" not in oof_df.columns:
        return None
    plt.figure(figsize=(10, 6))
    plt.scatter(oof_df["score"], oof_df[u_col], alpha=0.25, s=10)
    plt.title("Marketing Score vs EDL Uncertainty")
    plt.xlabel("score")
    plt.ylabel("uncertainty_edl")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    out = os.path.join(output_dir, "marketing_uncertainty_vs_score.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_marketing_topk_uncertainty(
    oof_df: pd.DataFrame,
    output_dir: str,
    top_k: int = 50,
) -> Optional[str]:
    u_col = "y_pred_uncertainty_edl"
    if u_col not in oof_df.columns or "score" not in oof_df.columns:
        return None
    top_df = oof_df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    plt.figure(figsize=(11, 6))
    plt.plot(np.arange(1, len(top_df) + 1), top_df[u_col].values, marker="o")
    plt.title(f"Marketing Top-{top_k} Score Rows: EDL Uncertainty")
    plt.xlabel("rank by score")
    plt.ylabel("uncertainty_edl")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    out = os.path.join(output_dir, "marketing_topk_uncertainty.png")
    plt.savefig(out)
    plt.close()
    return out


def export_uncertain_high_score(
    oof_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 50,
) -> Optional[str]:
    u_col = "y_pred_uncertainty_edl"
    out_path = os.path.join(output_dir, "uncertain_high_score.csv")

    if u_col not in oof_df.columns or "score" not in oof_df.columns:
        pd.DataFrame(columns=["score", "y_true", u_col]).to_csv(out_path, index=False)
        return out_path

    s_thr = float(oof_df["score"].quantile(0.90))
    u_thr = float(oof_df[u_col].quantile(0.90))
    out_df = oof_df[(oof_df["score"] >= s_thr) & (oof_df[u_col] >= u_thr)].copy()
    out_df = out_df.sort_values(["score", u_col], ascending=[False, False]).head(top_n)
    out_df.to_csv(out_path, index=False)
    return out_path

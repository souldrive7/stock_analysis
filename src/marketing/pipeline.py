import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime


from src.config import Config
from src.marketing.plots import (
    export_uncertain_high_score,
    plot_marketing_calibration_curve,
    plot_marketing_gain_chart,
    plot_marketing_shap_lgbm,
    plot_marketing_topk_uncertainty,
    plot_marketing_uncertainty_hist_edl,
    plot_marketing_uncertainty_vs_score,
)
from src.marketing.schemas import build_score_schema
from src.marketing.scoring import build_call_score
from src.marketing.utils import (
    detect_identifier_column,
    detect_target_column,
    encode_binary_target,
    fit_fold_preprocessor,
    transform_fold_features,
)
from src.models.edl import EDLModel
from src.models.lgbm import LGBMModel
from src.models.logistic import LogisticModel
from src.models.nn import NeuralNetModel


def _build_model_specs(enable_edl: bool) -> Dict[str, Tuple[type, dict]]:
    specs = {
        "lgbm": (LGBMModel, dict(Config.LGBM_PARAMS)),
        "logit": (LogisticModel, dict(Config.LOGISTIC_PARAMS)),
        "nn": (NeuralNetModel, dict(Config.NN_PARAMS)),
    }
    if enable_edl:
        specs["edl"] = (EDLModel, dict(Config.EDL_PARAMS))
    return specs


def _run_oof_for_model(
    model_cls: type,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Dict[str, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(X), dtype=float)
    oof_uncertainty = None

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr_raw = X.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X.iloc[va_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        prep = fit_fold_preprocessor(X_tr_raw)
        X_tr = transform_fold_features(X_tr_raw, prep)
        X_va = transform_fold_features(X_va_raw, prep)

        model = model_cls(params)
        model.train(X_tr, y_tr, X_va, y_va)
        oof_proba[va_idx] = model.predict(X_va)

        if hasattr(model, "predict_uncertainty"):
            unc = model.predict_uncertainty(X_va)
            if oof_uncertainty is None:
                oof_uncertainty = np.zeros(len(X), dtype=float)
            oof_uncertainty[va_idx] = unc

        print(f"[{model_cls.__name__}] fold {fold_idx}/{n_splits} done")

    return {"proba": oof_proba, "uncertainty": oof_uncertainty}

def run_basic_eda(
    df: pd.DataFrame,
    target: str,
    output_dir: str,
    positive_label=None,
    top_cat: int = 10,
) -> None:
    """
    Minimal EDA:
      - prints summary to console
      - saves text/csv files under output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    eda_txt = os.path.join(output_dir, f"eda_report_{ts}.txt")
    miss_csv = os.path.join(output_dir, f"eda_missing_{ts}.csv")
    dtypes_csv = os.path.join(output_dir, f"eda_dtypes_{ts}.csv")
    topcats_csv = os.path.join(output_dir, f"eda_top_categories_{ts}.csv")

    lines = []
    lines.append("=== EDA REPORT (minimal) ===")
    lines.append(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")
    lines.append(f"target: {target}")
    if positive_label is not None:
        lines.append(f"positive_label: {positive_label}")
    lines.append("")

    # head
    lines.append("--- head(5) ---")
    lines.append(df.head(5).to_string(index=False))
    lines.append("")

    # dtypes
    dtypes_df = df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ["column", "dtype"]
    dtypes_df.to_csv(dtypes_csv, index=False)
    lines.append("--- dtypes (saved) ---")
    lines.append(f"saved: {dtypes_csv}")
    lines.append("")

    # missing
    miss = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
    })
    miss["missing_rate"] = miss["missing_count"] / float(len(df))
    miss = miss.sort_values(["missing_rate", "missing_count"], ascending=False).reset_index(drop=True)
    miss.to_csv(miss_csv, index=False)
    lines.append("--- missing (top 30 shown; full saved) ---")
    lines.append(miss.head(30).to_string(index=False))
    lines.append(f"saved: {miss_csv}")
    lines.append("")

    # target distribution
    lines.append("--- target distribution ---")
    vc = df[target].value_counts(dropna=False)
    lines.append(vc.to_string())
    lines.append("")
    vcn = df[target].value_counts(normalize=True, dropna=False)
    lines.append("--- target distribution (rate) ---")
    lines.append(vcn.to_string())
    lines.append("")

    # numeric / categorical split (simple)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # numeric describe
    if len(num_cols) > 0:
        lines.append("--- numeric describe ---")
        lines.append(df[num_cols].describe().to_string())
        lines.append("")
    else:
        lines.append("--- numeric describe ---")
        lines.append("(no numeric columns)")
        lines.append("")

    # top categories
    top_rows = []
    for c in cat_cols:
        # skip target itself if it's categorical
        if c == target:
            continue
        vc = df[c].value_counts(dropna=False).head(top_cat)
        for k, v in vc.items():
            top_rows.append({"column": c, "value": str(k), "count": int(v)})

    topcats = pd.DataFrame(top_rows)
    if len(topcats) > 0:
        topcats.to_csv(topcats_csv, index=False)
        lines.append(f"--- top categorical values (saved; top {top_cat} per col) ---")
        lines.append(f"saved: {topcats_csv}")
        lines.append(topcats.head(50).to_string(index=False))
        lines.append("")
    else:
        lines.append("--- top categorical values ---")
        lines.append("(no categorical columns)")
        lines.append("")

    # print and save
    report = "\n".join(lines)
    print(report)
    with open(eda_txt, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[EDA] saved: {eda_txt}")

def plot_score_scatter_two_modes(
    oof_df: pd.DataFrame,
    output_dir: str,
    x_col: str,
    y_col: str,
    filename: str = "score_scatter_mean_vs_edl_weighted.png",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    d = oof_df[[x_col, y_col]].dropna()
    if len(d) == 0:
        print("[plot_score_scatter] No data to plot (all NaN).")
        return

    plt.figure()
    plt.scatter(d[x_col].values, d[y_col].values, s=8)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Score Scatter")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot_score_scatter] saved: {path}")

def run_marketing_pipeline(
    data_path: str,
    output_dir: str,
    target_col: str = None,
    n_splits: int = 5,
    random_state: int = 42,
    top_k: int = 200,
    include_debug_columns: bool = False,
    enable_edl: bool = False,
    score_mode: str = "mean_proba",
    run_eda: bool = False,
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"CSV not found: {data_path}\n"
            "Next action: place bank-full.csv under data/input/ "
            "or specify --data-path explicitly."
        )


    if run_eda:
        # TODO: Add EDA runner on leakage-dropped features (e.g., duration removed).
        print("run_eda=True: will run minimal EDA after loading and target detection.")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path, sep=";").reset_index(drop=True)
    print(f"loaded_csv={data_path}")
    print(f"shape={df.shape[0]} rows x {df.shape[1]} cols")

    target = detect_target_column(df, target_col)
    y, positive_label = encode_binary_target(df[target])
    print(f"target={target}, positive_label={positive_label}")
    if run_eda:
        # EDA should be executed on the raw df (before dropping columns) for transparency
        run_basic_eda(df=df, target=target, output_dir=output_dir, positive_label=positive_label)
    id_col = detect_identifier_column(df)
    if id_col:
        row_id = df[id_col]
    else:
        row_id = pd.Series(np.arange(len(df)), name="row_id")
    print(f"id_column={id_col if id_col else 'generated_row_id'}")

    feature_drop_cols = (
        [target]
        + ([id_col] if id_col else [])
        + (["duration"] if "duration" in df.columns else [])
    )
    X = df.drop(columns=feature_drop_cols)
    if "duration" in df.columns:
        print("leakage_guard=drop_feature:duration")
    else:
        print("leakage_guard=duration_not_found")
    prep_all = fit_fold_preprocessor(X)
    n_num = len(prep_all["num_cols"])
    n_cat = len(prep_all["cat_cols"])
    n_feat = len(prep_all["feature_columns"])
    print(f"preprocess_columns=num:{n_num}, cat:{n_cat}, transformed_dim:{n_feat}")

    oof_df = pd.DataFrame({"y_true": y})
    # Keep stable output schema even when EDL is disabled.
    oof_df["y_pred_proba_edl"] = np.nan
    oof_df["y_pred_uncertainty_edl"] = np.nan
    metrics_rows = []
    metric_lines: List[str] = []

    for model_name, (model_cls, params) in _build_model_specs(enable_edl=enable_edl).items():
        preds = _run_oof_for_model(
            model_cls=model_cls,
            params=params,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )
        oof_df[f"y_pred_proba_{model_name}"] = preds["proba"]
        if preds["uncertainty"] is not None:
            oof_df[f"y_pred_uncertainty_{model_name}"] = preds["uncertainty"]

        auc = roc_auc_score(y, preds["proba"])
        ap = average_precision_score(y, preds["proba"])
        metrics_rows.append({"model": model_name, "auc": auc, "ap": ap})
        metric_lines.append(f"{model_name}: AUC={auc:.6f}, AP={ap:.6f}")
        print(f"[{model_name}] AUC={auc:.6f}, AP={ap:.6f}")

    schema = build_score_schema()
    # 1) always compute mean_proba score
    score_mean = build_call_score(oof_df, schema, score_mode="mean_proba")
    oof_df["score_mean_proba"] = score_mean
    # 2) compute edl_uncertainty_weighted score only when EDL is enabled
    oof_df["score_edl_uncertainty_weighted"] = np.nan
    if enable_edl:
        try:
            oof_df["score_edl_uncertainty_weighted"] = build_call_score(
            oof_df, schema, score_mode="edl_uncertainty_weighted"
        )
        except Exception as e:
            print(f"[score] skip edl_uncertainty_weighted: {e}")
    # pipeline's main score column follows selected score_mode
    if score_mode == "mean_proba":
        oof_df["score"] = oof_df["score_mean_proba"]
    elif score_mode == "edl_uncertainty_weighted":
        oof_df["score"] = oof_df["score_edl_uncertainty_weighted"]
    else:
        # fallback: compute whatever requested
        oof_df["score"] = build_call_score(oof_df, schema, score_mode=score_mode)

    # scatter plot for comparison (only when EDL-weighted score exists)
    if oof_df["score_edl_uncertainty_weighted"].notna().any():
        plot_score_scatter_two_modes(
            oof_df=oof_df,
            output_dir=output_dir,
            x_col="score_mean_proba",
            y_col="score_edl_uncertainty_weighted",
        )
    else:
        print("[plot_score_scatter] skipped (no EDL-weighted score).")
    call_list = pd.DataFrame(
        {"row_id": row_id.values, "score": oof_df["score"].values, "__idx": np.arange(len(oof_df))}
    ).sort_values(["score", "__idx"], ascending=[False, True], kind="mergesort")
    call_list = call_list.head(top_k).reset_index(drop=True)
    # rank definition: 1..K order inside call_list_top200.csv
    call_list["rank"] = np.arange(1, len(call_list) + 1)

    if include_debug_columns:
        call_list["score_components"] = score_mode
        call_list["n_models_used"] = len(
            [c for c in oof_df.columns if c.startswith("y_pred_proba_")]
        )
        debug_cols = [c for c in oof_df.columns if c.startswith("y_pred_proba_")]
        if "y_pred_uncertainty_edl" in oof_df.columns:
            debug_cols.append("y_pred_uncertainty_edl")
        for col in debug_cols:
            call_list[col.replace("y_pred_", "")] = oof_df.loc[call_list["__idx"], col].values
    call_list = call_list.drop(columns=["__idx"])

    metrics_df = pd.DataFrame(metrics_rows).sort_values("auc", ascending=False).reset_index(drop=True)

    oof_path = os.path.join(output_dir, "oof_results.csv")
    metrics_path = os.path.join(output_dir, "metrics_summary.csv")
    call_path = os.path.join(output_dir, "call_list_top200.csv")
    summary_path = os.path.join(output_dir, "summary.txt")

    oof_df.to_csv(oof_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    call_list.to_csv(call_path, index=False)

    plot_marketing_gain_chart(oof_df, output_dir, include_score_curve=True)
    plot_marketing_calibration_curve(oof_df, output_dir)
    plot_marketing_shap_lgbm(X, y, output_dir, random_state=random_state)

    if oof_df["y_pred_uncertainty_edl"].notna().any():
        plot_marketing_uncertainty_hist_edl(oof_df, output_dir)
        plot_marketing_uncertainty_vs_score(oof_df, output_dir)
        plot_marketing_topk_uncertainty(oof_df, output_dir, top_k=min(top_k, 50))
    export_uncertain_high_score(oof_df, output_dir, top_n=50)

    summary_lines = [
        "Marketing Pipeline Summary",
        f"input_csv: {data_path}",
        f"input_shape: {df.shape[0]} rows x {df.shape[1]} cols",
        f"target_column: {target}",
        f"positive_label: {positive_label}",
        f"id_column: {id_col if id_col else 'generated_row_id'}",
        f"n_numeric_columns: {n_num}",
        f"n_categorical_columns: {n_cat}",
        f"transformed_feature_dim: {n_feat}",
        f"dropped_feature_for_leakage: {'duration' if 'duration' in df.columns else 'none'}",
        "rank_definition: rank is 1..K within call_list_top200.csv (score desc, input-order tie-break)",
        f"score_mode: {score_mode}",
        f"enable_edl: {enable_edl}",
        "",
        "Model Metrics",
        *metric_lines,
        "",
        "Output Files",
        f"- {oof_path}",
        f"- {metrics_path}",
        f"- {call_path}",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("=== Marketing pipeline done ===")
    print(f"target: {target} (positive label: {positive_label})")
    print(f"saved: {oof_path}")
    print(f"saved: {metrics_path}")
    print(f"saved: {call_path}")
    print(f"saved: {summary_path}")

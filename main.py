import sys
import os
import datetime

# パス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# 自作モジュールのインポート
from src.config import Config
from src.runner import Runner
from src.models.lgbm import LGBMModel
from src.models.logistic import LogisticModel
from src.models.nn import NeuralNetModel
from src.models.autoencoder import DenoisingAutoEncoder

# ========== データ生成関数 (高精度・安定版) ==========
def make_synthetic(n_weeks=104, n_customers=200, seed=7, lags=(1, 4, 12), out_dir="data/output"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    start_date = pd.to_datetime("2023-01-04")
    max_lag = max(lags)
    n_total_weeks = n_weeks + max_lag
    dates = pd.date_range(start_date, periods=n_total_weeks, freq="W-MON")

    # 市場データ
    market_data = []
    for d in dates:
        market_data.append({
            "date": d,
            "market_topix_return": rng.normal(0, 0.025),
            "market_sp500_return": rng.normal(0, 0.035),
            "market_japan_rate_change": rng.normal(0, 0.002),
            "market_us_rate_change": rng.normal(0, 0.01),
            "market_oil_return": rng.normal(0, 0.01),
        })
    market_df = pd.DataFrame(market_data)

    # 顧客データ
    customer_ids = [f"C{str(i).zfill(3)}" for i in range(1, n_customers + 1)]
    customer_data = []
    for cid in customer_ids:
        customer_data.append({
            "customer_id": cid,
            "customer_age": rng.integers(20, 75),
            "customer_asset_size": rng.choice(np.array([5, 10, 20, 50, 100])) * 1e6,
            "customer_risk_tolerance": rng.integers(1, 6),
            "sales_experience_years": rng.integers(1, 16),
            "customer_reported_assets": rng.choice(np.array([5, 10, 20, 50, 100])) * 1e6 * rng.uniform(0.8, 1.2)
        })
    customer_df = pd.DataFrame(customer_data)

    # 結合
    base_panel = pd.merge(
        customer_df.assign(key=1),
        market_df.assign(key=1),
        on="key"
    ).drop("key", axis=1)

    n_rows = len(base_panel)
    # ベースの行動量
    base_panel["sales_contact_count"] = rng.poisson(0.7, n_rows)
    base_panel["contact_visit"] = rng.binomial(1, 0.01, n_rows)
    
    # ダミー特徴量
    dummy_cols = [
        "contact_phone", "contact_email", "customer_trade_count", 
        "fundwrap_proposal_count", "fundwrap_buy_amount", "fundwrap_sell_amount",
        "portfolio_ratio_equity", "portfolio_ratio_fund", "portfolio_ratio_bond",
        "portfolio_ratio_insurance", "portfolio_ratio_fundwrap", "portfolio_diversification"
    ]
    for col in dummy_cols:
        base_panel[col] = rng.random(n_rows)

    df = base_panel.sort_values(["customer_id", "date"]).copy()

    # ラグ生成
    cols_to_lag = ["sales_contact_count", "contact_visit", "market_topix_return", "market_sp500_return"] + dummy_cols
    for lag in lags:
        for c in cols_to_lag:
            df[f"{c}_lag{lag}"] = df.groupby("customer_id")[c].shift(lag)

    # 目的変数生成
    def scale(col):
        std = col.std()
        if std == 0: std = 1.0
        return (col - col.mean()) / std

    logit_signal = (
        1.0 * scale(df["customer_risk_tolerance"]) +
        0.5 * scale(df["customer_asset_size"]) +
        0.8 * scale(df["sales_contact_count_lag1"].fillna(0)) +
        1.2 * scale(df["contact_visit_lag1"].fillna(0)) +
        0.6 * scale((df["market_topix_return_lag4"].abs() + df["market_sp500_return_lag4"].abs()).fillna(0))
    )
    
    noise = rng.normal(0, 0.2, size=len(df)) 
    base_logit = np.log(0.10 / 0.90)

    p = 1 / (1 + np.exp(-(base_logit + logit_signal + noise)))
    df["fundwrap_buy_flag_nextweek"] = rng.binomial(1, p)

    # クレンジング
    df = df.dropna(subset=[f"{c}_lag{max_lag}" for c in cols_to_lag])
    df["week_index"] = (df["date"] - df["date"].min()).dt.days // 7
    df = df.sort_values(["customer_id", "date"]).reset_index(drop=True)
    
    return df

# ========== AEパイプライン実行関数 (新規追加) ==========
def run_ae_pipeline(df, feature_cols, ae_params_config):
    """
    AutoEncoderの学習、特徴抽出、再構成誤差の計算を行い、
    特徴量を追加したDataFrameと新しいカラム名のリストを返す関数
    """
    print("\n--- Running AutoEncoder Feature Extraction ---")
    
    # (1) 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # (2) パラメータ設定 (input_dimを動的に設定)
    ae_params = ae_params_config.copy()
    ae_params['input_dim'] = X_scaled.shape[1]
    print(f"AE Params: {ae_params}")

    # (3) AEモデルの学習
    ae = DenoisingAutoEncoder(ae_params)
    ae.fit(X_scaled, X_val=X_scaled)
    
    # (4) 特徴量の抽出
    # A. 潜在変数 (Latent Features)
    ae_features = ae.transform(X_scaled)
    ae_col_names = [f'ae_feat_{i}' for i in range(ae_params['encoding_dim'])]
    df_ae = pd.DataFrame(ae_features, columns=ae_col_names, index=df.index)
    
    # B. 再構成誤差 (Reconstruction Error)
    reconstructed_data = ae.model.predict(X_scaled, verbose=0)
    mse = np.mean(np.power(X_scaled - reconstructed_data, 2), axis=1)
    df_ae['ae_recon_error'] = mse
    ae_col_names.append('ae_recon_error')
    
    print(f"Added Reconstruction Error feature. Mean MSE: {mse.mean():.4f}")
    
    # 結合
    df_out = pd.concat([df, df_ae], axis=1)
    
    print(f"Added {len(ae_col_names)} AE features.")
    
    return df_out, ae_col_names

# ========== 可視化関数群 ==========
def plot_gain_chart(oof_df, output_dir, run_id):
    print("\n--- Plotting Gain Chart ---")
    total_true = oof_df["y_true"].sum()
    if total_true == 0: return

    plt.figure(figsize=(10, 7))
    def plot_model_gain(col, label, color, style="-", width=1.5):
        if col not in oof_df.columns: return
        df_sorted = oof_df.sort_values(by=col, ascending=False)
        df_sorted["gain"] = df_sorted["y_true"].cumsum() / total_true
        df_sorted["percentile"] = np.arange(1, len(df_sorted) + 1) / len(df_sorted)
        plt.plot(df_sorted["percentile"], df_sorted["gain"], label=label, color=color, linestyle=style, linewidth=width)
    
    plot_model_gain("y_pred_proba_lgbm", "LightGBM", "blue")
    plot_model_gain("y_pred_proba_logit", "Logistic", "orange", "--")
    plot_model_gain("y_pred_proba_nn", "NN", "green", "-.")
    plot_model_gain("y_pred_proba_ensemble", "Ensemble", "red", "-", width=2.5)

    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Random")
    plt.plot([0, total_true/len(oof_df), 1], [0, 1, 1], linestyle=":", color="black", label="Ideal")
    plt.title(f"Gain Chart (Run: {run_id})")
    plt.xlabel("Percentage of Customers Contacted"); plt.ylabel("Cumulative Gain")
    plt.legend(); plt.grid(True, linestyle=':'); plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_id}_gain_chart.png")
    plt.close()

def plot_calibration_curve(oof_df, output_dir, run_id):
    print("\n--- Plotting Calibration Curve ---")
    if oof_df["y_true"].sum() == 0: return
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    
    def plot_model_calib(col, label, color, marker):
        if col not in oof_df.columns: return
        prob_true, prob_pred = calibration_curve(oof_df["y_true"], oof_df[col], n_bins=20, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker=marker, label=label, color=color)
        
    plot_model_calib("y_pred_proba_lgbm", "LightGBM", "blue", 'o')
    plot_model_calib("y_pred_proba_logit", "Logistic", "orange", 's')
    plot_model_calib("y_pred_proba_nn", "NN", "green", '^')
    plot_model_calib("y_pred_proba_ensemble", "Ensemble", "red", 'D')

    plt.title(f"Calibration Curve (Run: {run_id})")
    plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
    plt.legend(); plt.grid(True, linestyle=':'); plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_id}_calibration_curve.png")
    plt.close()

def plot_shap_summary(shap_values, X, output_dir, run_id):
    print("\n--- Plotting SHAP Summary ---")
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary (Run: {run_id})", y=1.05)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{run_id}_shap_summary.png", bbox_inches='tight')
    plt.close()

# ========== Main Execution ==========
def main():
    now = datetime.datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    print(f"=== Start Analysis (Run ID: {run_id}) ===\n")
    
    out_dir = f"data/output/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. データ生成
    df = make_synthetic(n_weeks=104, n_customers=200, out_dir=out_dir)
    
    # 基本特徴量の定義
    lag_cols = [c for c in df.columns if "_lag" in c]
    base_cols = ["customer_age", "customer_asset_size", "customer_risk_tolerance", "sales_experience_years"]
    feature_cols = base_cols + lag_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    target_col = 'fundwrap_buy_flag_nextweek'
    group_col = 'customer_id'

    # =========================================================
    # Phase 1.5: AutoEncoder Feature Extraction (Switchable)
    # =========================================================
    # Config.USE_AUTOENCODER が True の場合のみ実行
    if getattr(Config, "USE_AUTOENCODER", False):
        df, ae_new_cols = run_ae_pipeline(df, feature_cols, Config.AE_PARAMS)
        feature_cols += ae_new_cols
        print(f"AE Enabled: Total features increased to {len(feature_cols)}")
    else:
        print("\n--- AutoEncoder Feature Extraction Skipped (Config.USE_AUTOENCODER=False) ---")
    # =========================================================

    print(f"Final Features: {len(feature_cols)}, Target Rate: {df[target_col].mean():.2%}")
    results = df[['customer_id', target_col]].copy()
    results.rename(columns={target_col: 'y_true'}, inplace=True)
    
    # 2. モデル実行 (予測フェーズ)
    # --- LightGBM ---
    print("\n--- Running LightGBM ---")
    runner_lgbm = Runner(LGBMModel, Config.LGBM_PARAMS)
    lgbm_result = runner_lgbm.run_cv(df, feature_cols, target_col, group_col, return_shap=True)
    if isinstance(lgbm_result, tuple):
        results['y_pred_proba_lgbm'], shap_vals, X_shap = lgbm_result
        plot_shap_summary(shap_vals, X_shap, output_dir=out_dir, run_id=run_id)
    else:
        results['y_pred_proba_lgbm'] = lgbm_result
    
    # --- Logistic & NNs ---
    print("\n--- Running Logistic ---")
    results['y_pred_proba_logit'] = Runner(LogisticModel, Config.LOGISTIC_PARAMS).run_cv(df, feature_cols, target_col, group_col)
    
    print("\n--- Running NN ---")
    results['y_pred_proba_nn'] = Runner(NeuralNetModel, Config.NN_PARAMS).run_cv(df, feature_cols, target_col, group_col)

    # 3. アンサンブル
    print("\n--- Calculating Ensemble ---")
    weights = Config.ENSEMBLE_WEIGHTS
    pred_ensemble = np.zeros(len(results))
    total_w = 0
    
    if "lgbm" in weights:
        pred_ensemble += results['y_pred_proba_lgbm'] * weights["lgbm"]
        total_w += weights["lgbm"]
    if "nn" in weights:
        pred_ensemble += results['y_pred_proba_nn'] * weights["nn"]
        total_w += weights["nn"]
    
    if total_w > 0:
        results['y_pred_proba_ensemble'] = pred_ensemble / total_w
    else:
        results['y_pred_proba_ensemble'] = results['y_pred_proba_lgbm']

    # 4. 保存と可視化
    results.to_csv(f'{out_dir}/{run_id}_oof_results.csv', index=False)
    plot_gain_chart(results, output_dir=out_dir, run_id=run_id)
    plot_calibration_curve(results, output_dir=out_dir, run_id=run_id)
    
    # 5. Metrics
    scores = []
    y_true = results['y_true']
    target_models = ['LightGBM', 'Logistic', 'NN', 'Ensemble']
    target_cols = ['y_pred_proba_lgbm', 'y_pred_proba_logit', 'y_pred_proba_nn', 'y_pred_proba_ensemble']
    
    for model_name, col in zip(target_models, target_cols):
        if col in results.columns and not isinstance(results[col], int) and results[col].sum() != 0:
            try:
                auc = roc_auc_score(y_true, results[col])
                ap = average_precision_score(y_true, results[col])
                scores.append({'Model': model_name, 'AUC': auc, 'AP': ap})
            except ValueError:
                pass
    
    score_df = pd.DataFrame(scores)
    score_df.to_csv(f'{out_dir}/{run_id}_metrics_summary.csv', index=False)

    print(f"\n=== All Done. Results saved in: {out_dir} ===")
    print(score_df)

if __name__ == "__main__":
    main()

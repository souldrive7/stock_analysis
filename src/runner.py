import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import shap # SHAPライブラリ

class Runner:
    def __init__(self, model_cls, params, n_splits=5):
        self.model_cls = model_cls
        self.params = params
        self.n_splits = n_splits
    
    def run_cv(self, df, feature_cols, target_col, group_col, return_shap=False):
        """
        return_shap=True の場合、(oof_preds, shap_values, shap_data) を返す
        """
        X = df[feature_cols]
        y = df[target_col].values
        groups = df[group_col].values
        
        gkf = GroupKFold(n_splits=self.n_splits)
        oof_preds = np.zeros(len(df))
        scores = []
        
        # SHAP用リスト
        shap_vals_list = []
        X_va_list = []
        
        print(f"--- Running {self.model_cls.__name__} ---")
        
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_va = y[tr_idx], y[val_idx]
            
            # モデル生成と学習
            model = self.model_cls(self.params)
            model.train(X_tr, y_tr, X_va, y_va)
            
            # 予測
            preds = model.predict(X_va)
            oof_preds[val_idx] = preds
            
            # SHAP計算 (return_shap=Trueの場合のみ)
            if return_shap:
                try:
                    # モデル本体を取得
                    estimator = model.model
                    # Explainer作成
                    explainer = shap.TreeExplainer(estimator)
                    # 計算
                    shap_values = explainer.shap_values(X_va)
                    
                    # 2値分類の場合、shap_valuesはリスト([class0, class1])になることが多いので調整
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                        
                    shap_vals_list.append(shap_values)
                    X_va_list.append(X_va)
                except Exception as e:
                    print(f"  [Warning] SHAP calculation failed at Fold {fold+1}: {e}")
            
            # 評価
            if len(np.unique(y_va)) > 1:
                score = roc_auc_score(y_va, preds)
                scores.append(score)
                print(f"  Fold {fold+1} AUC: {score:.4f}")
            
        print(f"  Mean AUC: {np.mean(scores):.4f}\n")
        
        if return_shap and shap_vals_list:
            # 全FoldのSHAP値を結合して返す
            full_shap_vals = np.concatenate(shap_vals_list, axis=0)
            full_X_va = pd.concat(X_va_list, axis=0)
            return oof_preds, full_shap_vals, full_X_va
        
        return oof_preds
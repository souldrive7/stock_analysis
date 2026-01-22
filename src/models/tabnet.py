import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

class TabNetModel:
    def __init__(self, params):
        self.params = params
        self.model = None

    def train(self, tr_x, tr_y, va_x, va_y):
        # 修正: 入力がDataFrameならvaluesでnumpy化、すでにnumpyならそのまま使う
        tr_x = tr_x.values if hasattr(tr_x, 'values') else tr_x
        tr_y = tr_y.values if hasattr(tr_y, 'values') else tr_y
        va_x = va_x.values if hasattr(va_x, 'values') else va_x
        va_y = va_y.values if hasattr(va_y, 'values') else va_y

        # TabNetのパラメータ設定
        self.model = TabNetClassifier(
            n_d=self.params.get('n_d', 8),
            n_a=self.params.get('n_a', 8),
            n_steps=self.params.get('n_steps', 3),
            gamma=self.params.get('gamma', 1.3),
            n_independent=self.params.get('n_independent', 2),
            n_shared=self.params.get('n_shared', 2),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.params.get('lr', 2e-2)),
            scheduler_params=dict(step_size=50, gamma=0.9),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=0
        )

        self.model.fit(
            X_train=tr_x, y_train=tr_y, # .values を削除（変換済みのため）
            eval_set=[(va_x, va_y)],    # .values を削除
            eval_name=['valid'],
            eval_metric=['auc'],
            max_epochs=self.params.get('epochs', 100),
            patience=self.params.get('patience', 20),
            batch_size=self.params.get('batch_size', 1024),
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

    def predict(self, te_x):
        if self.model is None:
            return np.zeros(len(te_x))
        # 修正: 入力がDataFrameならnumpy化
        te_x = te_x.values if hasattr(te_x, 'values') else te_x
        return self.model.predict_proba(te_x)[:, 1]
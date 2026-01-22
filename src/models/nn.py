from .base import BaseModel
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

class NeuralNetModel(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        # 1. 欠損処理: Median
        self.imputer = SimpleImputer(strategy='median')
        # 2. RankGauss: 分布を強制的に正規分布へ
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)

    def _preprocess(self, X, is_train=True):
        if is_train:
            X_imp = self.imputer.fit_transform(X)
            return self.scaler.fit_transform(X_imp)
        else:
            X_imp = self.imputer.transform(X)
            return self.scaler.transform(X_imp)

    # --- Focal Loss 定義 ---
    def _focal_loss(self, gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) \
                   -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
        return focal_loss_fixed

    def train(self, X_train, y_train, X_val, y_val):
        X_tr = self._preprocess(X_train, True)
        X_va = self._preprocess(X_val, False)
        
        input_dim = X_tr.shape[1]
        
        # --- Modern NN Architecture ---
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        
        # 第1層
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('swish')) # ReLU -> Swish
        model.add(Dropout(0.3))
        
        # 第2層
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('swish'))
        model.add(Dropout(0.3))
        
        # 第3層
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('swish'))
        model.add(Dropout(0.2))
        
        model.add(Dense(1, activation='sigmoid'))
        
        # コンパイル (Focal Loss使用)
        model.compile(
            loss=self._focal_loss(
                gamma=self.params.get('gamma', 2.0),
                alpha=self.params.get('alpha', 0.25)
            ),
            optimizer=Adam(learning_rate=self.params.get('lr', 0.001)), 
            metrics=['AUC'] # 内部的にはAUC計算
        )
        
        es = EarlyStopping(monitor='val_AUC', mode='max', patience=10, restore_best_weights=True)
        
        # Focal Lossを使う場合、class_weightは通常設定しない（loss側で制御するため）
        model.fit(
            X_tr, y_train,
            validation_data=(X_va, y_val),
            epochs=self.params.get('epochs', 50),
            batch_size=self.params.get('batch_size', 64),
            callbacks=[es],
            verbose=0
        )
        self.model = model

    def predict(self, X):
        X_sc = self._preprocess(X, False)
        return self.model.predict(X_sc, verbose=0).flatten()
import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam

from .base import BaseModel


class EDLModel(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=params.get("seed", 42)
        )
        self.num_classes = 2
        self.annealing_epochs = max(1, int(params.get("annealing_epochs", 10)))
        self._kl_coeff = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def _preprocess(self, X, is_train: bool):
        if is_train:
            x_imp = self.imputer.fit_transform(X)
            return self.scaler.fit_transform(x_imp)
        x_imp = self.imputer.transform(X)
        return self.scaler.transform(x_imp)

    def _dirichlet_kl_to_uniform(self, alpha: tf.Tensor) -> tf.Tensor:
        beta = tf.ones_like(alpha)
        sum_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        sum_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        t1 = tf.math.lgamma(sum_alpha) - tf.reduce_sum(
            tf.math.lgamma(alpha), axis=1, keepdims=True
        )
        t2 = tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) - tf.math.lgamma(
            sum_beta
        )
        t3 = tf.reduce_sum(
            (alpha - beta) * (tf.math.digamma(alpha) - tf.math.digamma(sum_alpha)),
            axis=1,
            keepdims=True,
        )
        return t1 + t2 + t3

    def _edl_loss(self, y_true: tf.Tensor, alpha: tf.Tensor) -> tf.Tensor:
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_onehot = tf.one_hot(y_true, depth=self.num_classes)
        s = tf.reduce_sum(alpha, axis=1, keepdims=True)
        p = alpha / s
        mse = tf.reduce_sum(tf.square(y_onehot - p), axis=1, keepdims=True)
        var = tf.reduce_sum(
            alpha * (s - alpha) / (s * s * (s + 1.0)), axis=1, keepdims=True
        )
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
        kl = self._dirichlet_kl_to_uniform(alpha_tilde)
        return tf.reduce_mean(mse + var + self._kl_coeff * kl)

    def _build_model(self, input_dim: int) -> Model:
        inp = layers.Input(shape=(input_dim,))
        x = inp
        for units in self.params.get("hidden_layers", [128, 64, 32]):
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("swish")(x)
            x = layers.Dropout(self.params.get("dropout", 0.2))(x)
        evidence = layers.Dense(self.num_classes, activation="softplus")(x)
        return Model(inputs=inp, outputs=evidence)

    def train(self, X_train, y_train, X_val, y_val):
        x_tr = self._preprocess(X_train, True)
        x_va = self._preprocess(X_val, False)
        model = self._build_model(x_tr.shape[1])
        optimizer = Adam(learning_rate=self.params.get("lr", 1e-3))
        best_weights = model.get_weights()

        @tf.function
        def train_step(xb, yb):
            with tf.GradientTape() as tape:
                evidence = model(xb, training=True)
                alpha = evidence + 1.0
                loss = self._edl_loss(yb, alpha)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        @tf.function
        def eval_step(xb, yb):
            evidence = model(xb, training=False)
            alpha = evidence + 1.0
            return self._edl_loss(yb, alpha)

        best_val = np.inf
        wait = 0
        epochs = int(self.params.get("epochs", 50))
        bs = int(self.params.get("batch_size", 256))
        patience = int(self.params.get("patience", 10))

        tr_ds = (
            tf.data.Dataset.from_tensor_slices((x_tr.astype(np.float32), y_train.astype(np.int32)))
            .shuffle(len(x_tr))
            .batch(bs)
        )
        va_ds = tf.data.Dataset.from_tensor_slices(
            (x_va.astype(np.float32), y_val.astype(np.int32))
        ).batch(bs)

        for ep in range(epochs):
            self._kl_coeff.assign(min(1.0, (ep + 1) / float(self.annealing_epochs)))
            for xb, yb in tr_ds:
                train_step(xb, yb)

            val_losses = [float(eval_step(xb, yb).numpy()) for xb, yb in va_ds]
            val_loss = float(np.mean(val_losses))
            if val_loss < best_val:
                best_val = val_loss
                best_weights = model.get_weights()
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break

        model.set_weights(best_weights)
        self.model = model

    def _predict_alpha(self, X) -> np.ndarray:
        x_sc = self._preprocess(X, False)
        evidence = self.model.predict(x_sc, verbose=0)
        return evidence + 1.0

    def predict(self, X) -> np.ndarray:
        alpha = self._predict_alpha(X)
        denom = np.sum(alpha, axis=1, keepdims=True)
        denom = np.where(denom <= 0, 1.0, denom)
        return alpha[:, 1] / denom[:, 0]

    def predict_uncertainty(self, X) -> np.ndarray:
        alpha = self._predict_alpha(X)
        denom = np.sum(alpha, axis=1)
        denom = np.where(denom <= 0, 1.0, denom)
        uncertainty = float(self.num_classes) / denom
        return np.clip(uncertainty, 0.0, 1.0)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class DenoisingAutoEncoder:
    def __init__(self, params: dict):
        """
        特徴抽出用のDenoising AutoEncoder
        params: Config.AE_PARAMS を受け取る
        """
        self.params = params
        self.input_dim = params['input_dim']
        self.encoding_dim = params['encoding_dim']
        self.learning_rate = params.get('learning_rate', 0.001)
        self.model = None
        self.encoder = None
        
        self._build_model()

    def _build_model(self):
        # 1. Input Layer
        input_layer = Input(shape=(self.input_dim,))
        
        # 2. Gaussian Noise (Denoising)
        # 入力にノイズを加えることで、モデルが「本質的なパターン」だけを学習するように促す
        # 株価データのようにノイズが多いデータセットで非常に有効です
        noisy_input = GaussianNoise(0.05)(input_layer)

        # 3. Encoder (圧縮プロセス)
        x = Dense(64, activation='relu')(noisy_input)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # 4. Latent Space (ここが抽出したい特徴量)
        latent_space = Dense(self.encoding_dim, activation='relu', name='latent_space')(x)

        # 5. Decoder (復元プロセス)
        x = Dense(32, activation='relu')(latent_space)
        x = Dense(64, activation='relu')(x)
        # 入力と同じ次元に戻す（標準化済みデータを想定するため活性化関数はlinear）
        decoded = Dense(self.input_dim, activation='linear')(x)

        # モデル構築
        self.model = Model(inputs=input_layer, outputs=decoded)
        self.encoder = Model(inputs=input_layer, outputs=latent_space)
        
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

    def fit(self, X_train, X_val=None):
        """
        学習を実行するメソッド
        """
        callbacks = []
        if 'patience' in self.params:
            callbacks.append(EarlyStopping(
                monitor='val_loss', 
                patience=self.params['patience'], 
                restore_best_weights=True
            ))
        
        validation_data = (X_val, X_val) if X_val is not None else None
        
        # AEは入力と出力が同じになるように学習する（教師なし学習）
        self.model.fit(
            X_train, X_train, 
            epochs=self.params.get('epochs', 50),
            batch_size=self.params.get('batch_size', 256),
            shuffle=True,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

    def transform(self, X):
        """
        データを入力して、潜在特徴量（AE特徴量）を出力する
        """
        return self.encoder.predict(X, verbose=0)
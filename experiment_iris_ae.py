import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# 乱数固定（再現性のため）
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def run_iris_experiment():
    print("=== Irisデータ読み込みと前処理 ===")
    # 1. データ準備
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names

    # Pandas DataFrameにして確認しやすくする
    df = pd.DataFrame(X, columns=feature_names)
    print(f"データ形状: {df.shape}")
    
    # 標準化 (Autoencoderには必須)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 訓練/テスト分割 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED
    )

    print("\n=== Autoencoderモデル構築 ===")
    # 2. モデル定義
    input_dim = X_train.shape[1] # 4次元
    encoding_dim = 2             # 2次元に圧縮（可視化のため）

    # Input Layer
    input_layer = Input(shape=(input_dim,))

    # Encoder (特徴抽出)
    # 4 -> 3 -> 2 という段階的な圧縮
    encoded = Dense(3, activation='relu')(input_layer)
    encoded = BatchNormalization()(encoded) # 学習安定化
    latent_space = Dense(encoding_dim, activation='linear', name='latent_space')(encoded)

    # Decoder (復元)
    # 2 -> 3 -> 4
    decoded = Dense(3, activation='relu')(latent_space)
    decoded = Dense(input_dim, activation='linear')(decoded)

    # モデル結合
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder_model = Model(inputs=input_layer, outputs=latent_space)

    # コンパイル
    autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    
    # 構造表示
    autoencoder.summary()

    print("\n=== 学習開始 ===")
    # 3. 学習
    history = autoencoder.fit(
        X_train, X_train,
        epochs=100,
        batch_size=16,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=1
    )

    print("\n=== 分析と可視化 ===")
    # 4. 可視化設定
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(15, 6))

    # --- Plot A: 学習曲線 ---
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Learning Curve (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # --- Plot B: 潜在空間の分布 (次元圧縮結果) ---
    # 4次元データを2次元に落とし込んだ結果
    X_encoded = encoder_model.predict(X_scaled)
    
    ax2 = fig.add_subplot(1, 3, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 青, 橙, 緑
    
    for i, target_name in enumerate(target_names):
        ax2.scatter(
            X_encoded[y == i, 0], 
            X_encoded[y == i, 1], 
            c=colors[i], 
            label=target_name, 
            alpha=0.7, 
            edgecolor='k'
        )
    
    ax2.set_title('Latent Space (2D Compression)')
    ax2.set_xlabel('Latent Dim 1')
    ax2.set_ylabel('Latent Dim 2')
    ax2.legend()

    # --- Plot C: 再構成誤差 (異常検知の視点) ---
    # 元のデータと復元データの差分を見る
    X_pred = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    
    ax3 = fig.add_subplot(1, 3, 3)
    for i, target_name in enumerate(target_names):
        sns.kdeplot(mse[y == i], ax=ax3, fill=True, label=target_name, color=colors[i])
    
    ax3.set_title('Reconstruction Error Distribution')
    ax3.set_xlabel('MSE Error')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    print("完了しました。グラフウィンドウを確認してください。")

if __name__ == "__main__":
    run_iris_experiment()
class Config:
    N_SPLITS = 5
    SEED = 42

    # アンサンブルの重み設定
    ENSEMBLE_WEIGHTS = {
        "lgbm": 0.5,
        "nn": 0.25,
        "simple_nn": 0.25
    }

    # === LightGBM Parameters ===
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "is_unbalance": False,
        "n_estimators": 1000,
        "early_stopping_rounds": 50
    }

    # === Logistic Regression Parameters ===
    LOGISTIC_PARAMS = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 5000,
        'random_state': 42
    }

    # === Modern NN Parameters ===
    NN_PARAMS = {
        'hidden_layers': [128, 64, 32],
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 256,
        'patience': 10
    }
    
    # === Simple NN Parameters ===
    SIMPLE_NN_PARAMS = {
        'input_dim': None,
        'hidden_dim': 64,
        'output_dim': 1,
        'learning_rate': 0.01,
        'epochs': 50,
        'batch_size': 128
    }

    # === AutoEncoder Parameters ===
    # 特徴抽出用の設定です
    # BaseModel を継承せず、独立した「特徴抽出器」として実装。
    # これにより Runner のロジック（教師あり学習用）と混ざるのを防ぐ。
    AE_PARAMS = {
        'input_dim': None,        # 実行時に決定
        'encoding_dim': 16,       # 圧縮後の次元数（抽出したい特徴量数）
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 256,
        'patience': 5             # Early Stopping用
    }
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
        "is_unbalance": False, # データ生成が綺麗なのでFalseでも十分精度が出ます
        "n_estimators": 1000,
        "early_stopping_rounds": 50
    }

    # === Logistic Regression Parameters ===
    # 収束するように最適化された設定
    LOGISTIC_PARAMS = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',  # 高速で安定したソルバー
        'max_iter': 5000,   # 警告が出ないよう十分な回数を確保
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
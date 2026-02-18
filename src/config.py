class Config:
    N_SPLITS = 5
    SEED = 42

    # アンサンブルの重み設定
    ENSEMBLE_WEIGHTS = {
        "lgbm": 0.5,
        "nn": 0.25,
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

    # === NN Parameters ===
    NN_PARAMS = {
        'hidden_layers': [128, 64, 32],
        'dropout': 0.3,
        'lr': 0.001,
        'epochs': 50,
        'batch_size': 256,
        'patience': 10,
        'seed': 42,
    }

    # === EDL Parameters ===
    EDL_PARAMS = {
        "hidden_layers": [128, 64, 32],
        "dropout": 0.3,
        "lr": 0.001,
        "epochs": 50,
        "batch_size": 256,
        "patience": 10,
        "annealing_epochs": 10,
        "seed": 42,
    }

# === Feature Engineering Flags ===
    USE_AUTOENCODER = True 

# === AutoEncoder Parameters ===
    AE_PARAMS = {
        'input_dim': None,
        'encoding_dim': 4,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 256,
        'patience': 5
        'seed': 42,
    }

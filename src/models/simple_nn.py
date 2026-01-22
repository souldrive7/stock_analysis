import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class SimpleNeuralNetModel:
    def __init__(self, params):
        self.params = params
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # メソッド名を fit から train に変更しました
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # 1. シンプルなスケーリング (RankGaussではなく標準化)
        if hasattr(tr_x, "values"):
            tr_x = tr_x.values
        if va_x is not None and hasattr(va_x, "values"):
            va_x = va_x.values
            
        tr_x = self.scaler.fit_transform(tr_x)
        if va_x is not None:
            va_x = self.scaler.transform(va_x)

        # Tensor化
        tr_x_t = torch.tensor(tr_x, dtype=torch.float32).to(self.device)
        
        # yがSeriesの場合に対応
        if hasattr(tr_y, "values"):
            tr_y = tr_y.values
        tr_y_t = torch.tensor(tr_y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # DataLoader
        batch_size = self.params.get("batch_size", 64)
        train_loader = DataLoader(TensorDataset(tr_x_t, tr_y_t), batch_size=batch_size, shuffle=True)

        # 2. シンプルなネットワーク定義 (ReLU使用, Swish不使用)
        input_dim = tr_x.shape[1]
        hidden_dim = self.params.get("hidden_units", [64, 32])
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU()) # ここをReLUに固定
            layers.append(nn.Dropout(0.2))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers).to(self.device)

        # 3. シンプルなロス関数 (Focal LossではなくBCE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.get("lr", 1e-3))
        
        # 学習ループ
        epochs = self.params.get("epochs", 20)
        for epoch in range(epochs):
            self.model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

    def predict(self, te_x):
        self.model.eval()
        if hasattr(te_x, "values"):
            te_x = te_x.values
            
        te_x = self.scaler.transform(te_x)
        te_x_t = torch.tensor(te_x, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(te_x_t)
        
        return preds.cpu().numpy().ravel()
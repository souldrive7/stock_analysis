import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=192, n_blocks=3, n_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None: d_ff = d_token * 2
        
        self.feature_tokenizer = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(n_features)
        ])
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        tokens = []
        for i, layer in enumerate(self.feature_tokenizer):
            tokens.append(layer(x[:, i:i+1].unsqueeze(-1)).squeeze(-2))
        x_emb = torch.stack(tokens, dim=1) 
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        
        out = self.transformer(x_emb)
        cls_out = out[:, 0, :]
        return self.head(cls_out)

class TransformerModel:
    def __init__(self, params):
        self.params = params
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, tr_x, tr_y, va_x, va_y):
        # 修正: 入力がDataFrameならnumpy化、すでにnumpyならそのまま使う
        tr_x = tr_x.values if hasattr(tr_x, 'values') else tr_x
        tr_y = tr_y.values if hasattr(tr_y, 'values') else tr_y
        va_x = va_x.values if hasattr(va_x, 'values') else va_x
        va_y = va_y.values if hasattr(va_y, 'values') else va_y

        n_features = tr_x.shape[1]
        self.model = FTTransformer(
            n_features=n_features,
            d_token=self.params.get('d_token', 64),
            n_blocks=self.params.get('n_blocks', 2),
            n_heads=self.params.get('n_heads', 4),
            dropout=self.params.get('dropout', 0.1)
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.get('lr', 1e-3), weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        # Tensor変換時も .values を削除
        train_ds = TensorDataset(torch.tensor(tr_x, dtype=torch.float32), torch.tensor(tr_y, dtype=torch.float32))
        valid_ds = TensorDataset(torch.tensor(va_x, dtype=torch.float32), torch.tensor(va_y, dtype=torch.float32))
        train_dl = DataLoader(train_ds, batch_size=self.params.get('batch_size', 512), shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size=self.params.get('batch_size', 512), shuffle=False)
        
        best_loss = float('inf')
        patience = self.params.get('patience', 10)
        counter = 0
        
        for epoch in range(self.params.get('epochs', 50)):
            self.model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb).squeeze()
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in valid_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = self.model(xb).squeeze()
                    val_loss += criterion(out, yb).item() * xb.size(0)
            val_loss /= len(valid_ds)
            
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                self.best_state = self.model.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    break
        
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)

    def predict(self, te_x):
        if self.model is None:
             return np.zeros(len(te_x))
        # 修正: 入力がDataFrameならnumpy化
        te_x = te_x.values if hasattr(te_x, 'values') else te_x
        
        self.model.eval()
        ds = TensorDataset(torch.tensor(te_x, dtype=torch.float32)) # .values削除
        dl = DataLoader(ds, batch_size=512, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, in dl:
                xb = xb.to(self.device)
                out = self.model(xb).squeeze()
                preds.append(torch.sigmoid(out).cpu().numpy())
        return np.concatenate(preds)
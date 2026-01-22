from .base import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer # NNとは違い、平均や中央値で単純に埋める想定

class LogisticModel(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.scaler = RobustScaler()
        # Logitも欠損には弱いので埋める
        self.imputer = SimpleImputer(strategy='mean')

    def train(self, X_train, y_train, X_val, y_val):
        X_tr = self.scaler.fit_transform(self.imputer.fit_transform(X_train))
        
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_tr, y_train)

    def predict(self, X):
        X_sc = self.scaler.transform(self.imputer.transform(X))
        return self.model.predict_proba(X_sc)[:, 1]
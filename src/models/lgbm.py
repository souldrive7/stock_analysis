from .base import BaseModel
import lightgbm as lgb

class LGBMModel(BaseModel):
    def train(self, X_train, y_train, X_val, y_val):
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
        self.model = lgb.train(self.params, dtrain, valid_sets=[dvalid], callbacks=callbacks, num_boost_round=1000)

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)
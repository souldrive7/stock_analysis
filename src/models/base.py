from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self, params: dict):
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
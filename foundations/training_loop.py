import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        b= 0
        w = np.zeros((X.shape[1]))
        n = len(y)
        print(X.shape[0], X.shape[1])
        for epoch in range(epochs):
            y_hat = X @ w + b
            loss = np.mean(y_hat-y)**2
            dl_dw = 2/n * X.T @ (y_hat-y)
            dl_db = 2/n * np.sum(y_hat-y)
            w = w - lr*dl_dw
            b = b - lr*dl_db
        w = np.round(w, decimals= 5)
        b = np.round(b, decimals= 5)
        return w,b
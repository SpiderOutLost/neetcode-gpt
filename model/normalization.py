import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(
        self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        denom = np.var(x) + 10 ** (-5)
        x = (x - np.mean(x)) / denom**0.5
        x= x*gamma + beta
        return np.round(x,5)

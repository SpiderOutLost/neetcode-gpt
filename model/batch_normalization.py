import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        if training:
            x = np.array(x)
            x_new = (x - np.mean(x, axis=0))/ np.sqrt(np.var(x,axis=0) + eps)
            target = gamma*x_new + beta
            running_mean = (1-momentum) * running_mean + momentum * np.mean(x, axis= 0)
            running_var = (1-momentum) * running_var + momentum * np.var(x, axis= 0)
        else:
            x_new = (x - running_mean)/ np.sqrt(running_var+eps)
            target = gamma*x_new + beta
        rounded_data = [np.round(arr, decimals= 4) for arr in [target, running_mean, running_var]]
        target, running_mean, running_var = rounded_data
        return target, running_mean, running_var
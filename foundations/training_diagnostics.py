import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        print(x)
        with torch.no_grad():
            for module in model.children():
                x = module(x) # прямой ход
                if isinstance(module, nn.Linear): # проверка на соответствие линейному слою
                    mean_val= round(x.mean().item(), 4)
                    std_val= round(x.std().item(), 4)
                    if x.dim() >= 2:
                        dead_frac= round(((x<=0).all(dim=0)).float().mean().item(),4)
                    else:
                        dead_frac = round((x <= 0).float().mean().item(), 4)
                    stats.append({
                        "mean": mean_val, "std": std_val, "dead_fraction": dead_frac
                    })
        return stats
    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
            model.zero_grad()
            stats = []
            predictions = model(x)
            criterion = nn.MSELoss()
            loss = criterion(predictions, y)
            loss.backward()
            for module in model.children():
                if isinstance(module, nn.Linear):
                    grad = module.weight.grad
                    grad_mean = torch.round(torch.mean(grad), decimals= 4)
                    grad_std = torch.round(torch.std(grad), decimals= 4)
                    grad_norm = torch.round(torch.norm(grad), decimals= 4)
                    stats.append({
                        "mean": grad_mean.item(),
                        "std": grad_std.item(),
                        "norm": grad_norm.item()
                    })
            return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:

        for s in activation_stats:
            if s["dead_fraction"] > 0.5:
                return "dead_neurons"
        for s in gradient_stats:
            if s["norm"]>1000:
                return "exploding_gradients"
        if gradient_stats and gradient_stats[-1]["norm"]<=1e-5:
            return "vanishing_gradients"
        for s in activation_stats:
            if s["std"] < 0.1:
                return "vanishing_gradients"
            elif s["std"] > 10.0:
                return "exploding_gradients"
        return "healthy"
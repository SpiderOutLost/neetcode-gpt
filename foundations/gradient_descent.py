class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        if iterations == 0:
            return init
        x_old = init
        while iterations > 0:
            x_new = x_old - learning_rate*2*x_old
            x_old = x_new
            iterations -= 1
        return round(x_new, 5)
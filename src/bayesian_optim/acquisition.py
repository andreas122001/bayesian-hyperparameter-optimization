from abc import ABC, abstractmethod
from botorch.models.gpytorch import GPyTorchModel
import torch


class AcquisitionFunction(ABC):
    def __init__(self, model: GPyTorchModel) -> None:
        super().__init__()
        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.acquire(x)

    @abstractmethod
    def acquire(self, x: torch.Tensor) -> torch.Tensor:
        pass


class CustomExpectedImprovement(AcquisitionFunction):
    """
    Implementation of a custom Expected Improvement (EI) function including a exploration parameter ξ.

    :param model: the trained gaussian process model
    :param best_y: the best observed value
    :param ksi: parameter for controlling exploration-exploitation trade-off. Higher Ksi (ξ) increases exploration.
    """

    def __init__(
        self, model: GPyTorchModel, best_y: torch.Tensor, ksi: float = 0.0
    ) -> None:
        super().__init__(model)
        self.best_y = best_y
        self.ksi = ksi

    def acquire(self, x: torch.Tensor) -> torch.Tensor:
        # See https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html

        # Definition:
        # EI(x) = (μ(x) - f(x*) − ξ) * Φ((μ(x) - f(x*) - ξ)/σ(x)) + σ(x) * φ((μ(x) - f(x*) - ξ)/σ(x))

        posterior = self.model.posterior(x)
        (mean, var) = (posterior.mean, posterior.variance)
        std = var.sqrt()

        improvement = mean - self.best_y - self.ksi  # raw improvement
        gamma = improvement / std  # standard improvement

        normal = torch.distributions.Normal(0.0, 1.0)
        cdf = normal.cdf(gamma)
        pdf = normal.log_prob(gamma).exp()

        ei = (improvement * cdf) + (std * pdf)

        return ei.reshape(-1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from gaussian_process import GaussianProcess
    from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement

    x_train = torch.tensor([[0.2, 0.4], [0.4, 0.3], [0.9, 0.1]])
    y_train = torch.tensor([[1.0], [0.8], [0.3]])
    model: GPyTorchModel = GaussianProcess().fit(x_train, y_train)

    x = torch.linspace(0, 1, 100)
    grid = torch.stack([x, x], dim=-1).unsqueeze(1)
    model.posterior(grid)

    ei1 = CustomExpectedImprovement(model, y_train.max(), ksi=1.0)
    ei2 = ExpectedImprovement(model, y_train.max())

    ei_val1 = ei1(grid).detach()
    ei_val2 = ei2(grid).detach()
    plt.plot(ei_val1, alpha=1.0)
    plt.plot(ei_val2, "--", alpha=0.9)
    plt.show()

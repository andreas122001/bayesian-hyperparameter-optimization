import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections.abc import Callable
from tqdm import tqdm

from src.bayesian_optim.gaussian_process import GaussianProcess
from src.bayesian_optim.acquisition import CustomExpectedImprovement


class BayesianOptimizer:
    """
    Estimates the maximum point of the given function via a Gaussian Process (GP) and Expected Improvement (EI) acquisition.

    :param objective_f: the function to estimate.
    :param ksi: exploration parameter for the acquisition function, where a higher value favors exploration over explitation.
    """

    def __init__(
        self, objective_f: Callable, min_bound, max_bound, ksi=0.01, use_log_scale=True
    ) -> None:
        self.objective_f = objective_f
        self.ksi = ksi
        self.use_log = use_log_scale

        self.gp = GaussianProcess()

        self.sobol_sampler = torch.quasirandom.SobolEngine(1, scramble=True)

        self.bounds = torch.tensor([min_bound, max_bound])
        if self.use_log:
            self.bounds = self.bounds.log10()
            self.grid = torch.logspace(*self.bounds, steps=100, base=10)
        else:
            self.grid = torch.linspace(*self.bounds, steps=100)

        self.grid = self.grid.unsqueeze(-1).unsqueeze(-1)

        self.train_x = torch.tensor([])
        self.train_y = torch.tensor([])

    def step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one step of Gaussian Proccess prediction and finds and predicts the next point to sample \
        via Expected Improvement. The next point to sample is sampled and added to the training set. Returns \
        the predicted mean, standard deviation and expected improvement values for analysis and/or visualization.

        :returns: the mean, standard deviation, and expected improvement values as a tuple of tensors. 
        """
        gp_model = self.gp.fit(
            self.train_x.unsqueeze(-1),
            self.train_y.unsqueeze(-1),
        )

        # Calculate the posterior mean and std for visualization
        posterior = gp_model.posterior(self.grid)
        mean = posterior.mean.detach().squeeze(-1).squeeze(-1)
        std = posterior.variance.sqrt().detach().squeeze(-1).squeeze(-1)

        # Calculate expected improvement and next sample
        # Add a bit more exploration with ksi=0.01
        ei = CustomExpectedImprovement(
            model=gp_model, best_y=self.train_y.max(), ksi=self.ksi
        )
        ei_val = ei.acquire(self.grid)

        next_x = self.grid[ei_val.argmax()][0].double()
        next_y = self.objective_f(next_x)
        self._add_datapoint(next_x, next_y)

        return mean, std, ei_val

    def initialize(self, n_samples: int) -> None:
        """
        Initializes the internal training set with random points from a Sobol sequence. \
        This is further used by the Gaussian Process to estimate the objective in the :meth:`step`-method.

        :param n_samples: how many samples to initialize.
        """
        sampled_x = self.sobol_sampler.draw(n_samples, dtype=torch.float64)

        # Convert Sobol sequence to log scale
        if self.use_log:
            sampled_x = self.bounds[0] + sampled_x * (self.bounds[1] - self.bounds[0])
            sampled_x = 10**sampled_x

        for next_x in tqdm(sampled_x, desc="Initializing", leave=False):
            next_y = self.objective_f(next_x)
            self._add_datapoint(next_x, next_y)

    def visualize(
        self, mean: torch.Tensor, std: torch.Tensor, ei_val: torch.Tensor
    ) -> Figure:
        """
        Creates a matplotlib figure of the current mean and standard deviation of the Gaussian Process, \
        in addition to the current Expected Improvement values.

        :param mean: the current predicted mean.
        :param std: the current predicted standard deviation.
        :param ei_val: the current expected improvement values.
        """
        best_x, _ = self.get_current_max_point(mean)

        plt.figure(figsize=(9, 8))

        plt.subplot(2, 1, 1)
        plt.title("Predicted objective")
        plt.plot(self.grid[:, 0, 0], mean, label="Mean")
        if self.use_log:
            plt.xscale("log")
        plt.axvline(
            best_x.item(),
            linestyle="dashed",
            label="Current best",
        )
        plt.text(best_x * 1.1, mean.min(), s=f"x={best_x:.1e}")
        plt.fill_between(
            self.grid[:, 0, 0],
            mean - 2 * std,
            mean + 2 * std,
            alpha=0.2,
            color="orange",
            label="Std. dev.",
        )
        plt.scatter(
            self.train_x[:-1], self.train_y[:-1], color="red", label="Observations"
        )
        plt.scatter(
            self.train_x[-1:], self.train_y[-1:], color="green", label="Next sample"
        )
        plt.xlabel("Learning rate")
        plt.ylabel("Accuracy")
        plt.legend(loc=3, framealpha=0.6)

        plt.subplot(2, 1, 2)
        plt.title("Acquisition function")
        plt.plot(self.grid[:, 0, 0], ei_val.detach())
        if self.use_log:
            plt.xscale("log")
        plt.scatter(
            self.grid[ei_val.argmax(0), 0, 0], ei_val.detach().max(), color="green"
        )
        plt.xlabel("Learning rate")
        plt.ylabel("Expected improvement")
        plt.tight_layout()

        return plt.gcf()

    def get_current_max_point(
        self, mean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the xy-coordinate of the maximum of the predicted mean.

        :returns: the x and y values as a tuple of tensors.
        """
        return self.grid[mean.argmax(), 0, 0], mean.max()

    def _add_datapoint(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Adds the xy-coordinate to the training set.

        :param x: the x coord.
        :param y: the y coord.
        """
        self.train_x = torch.cat([self.train_x, x])
        self.train_y = torch.cat([self.train_y, y])


if __name__ == "__main__":
    objective = lambda x: 2 * torch.sin(3.14 * 1.2 * x) + 0.2 * torch.cos(3.14 * 12 * x)
    optimizer = BayesianOptimizer(objective_f=objective)
    optimizer.initialize(3)

    for _ in range(7):
        mean, std, ei = optimizer.step()
        best_x, best_y = optimizer.get_current_max_point(mean)
        fig = optimizer.visualize(mean, std, ei)
        print(f"Best point: [{best_x.item():.3f}, {best_y.item():.3f}]")
        plt.show()

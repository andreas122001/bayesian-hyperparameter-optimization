import torch

from typing import Optional

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize  # might need
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior

from botorch.models.gpytorch import GPyTorchModel


class GaussianProcess:
    """
    A wrapper class for the gaussian process.
    """

    def __init__(self, bounds: Optional[torch.Tensor] = None) -> None:
        self.bounds = bounds

    def fit(self, train_x, train_y) -> GPyTorchModel:
        """
        Fits the Gaussian Process to the input training set.

        :param train_x: the training x samples
        :param train_y: the training y observations
        :returns: the trained Gaussian Process model.
        """
        if self.bounds is not None:
            train_x = normalize(train_x, self.bounds)

        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            likelihood=GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0)),
        )
        mll = ExactMarginalLogLikelihood(
            model.likelihood, model
        )  # could maybe also be parameterized
        fit_gpytorch_mll(mll, approx_mll=True)

        return model

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize  # might need
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior

from botorch.posteriors.gpytorch import GPyTorchPosterior


class GaussianProcess:
    def __init__(self) -> None:
        pass

    def fit(self, train_x, train_y) -> GPyTorchPosterior:
        likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.0, 10.0))
        model = SingleTaskGP(
            train_X=train_x.unsqueeze(-1).double(),
            train_Y=train_y.unsqueeze(-1).double(),
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, approx_mll=True)

        return model


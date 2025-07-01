import torch
import os
import matplotlib.pyplot as plt

import argparse
from argparse import Namespace
from tqdm import tqdm

from src.bayesian_optim.bayesian_optimizer import BayesianOptimizer
from src.dl.trainer import FashionMNISTTrainer, Hyperparameters
from src.dl.resnet import CustomResNet

import logging

logger = logging.getLogger(__name__)


def _create_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="Bayesian Hyperparameter Optimizer",
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--init-steps",
        type=int,
        default=3,
        help="How many steps to initialize the bayesian optimizer with.",
    )
    parser.add_argument(
        "--optim-steps",
        type=int,
        default=7,
        help="How many steps to perform Bayesian Optimization for.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Displays the figure interactively for each iteration.",
    )
    parser.add_argument(
        "-s",
        "--do-save",
        action="store_true",
        help="Saves the figures under './figures/'.",
    )

    return parser.parse_args()


def main(args: Namespace) -> None:
    model = CustomResNet()
    ml_trainer = FashionMNISTTrainer(model)

    def objective(
        lr: torch.Tensor,
    ):  # could potentially be changed to select which param to optimize for
        torch.manual_seed(0)  # for more stable BO estimate
        hyperparams = Hyperparameters(
            epochs=2,
            learning_rate=lr.item(),
            batch_size=96,
        )
        # Train, test and return the accuracy of the ResNet model
        return (
            ml_trainer.train_and_test(hparams=hyperparams)
            .unsqueeze(0)
            .detach()
            .cpu()
            .double()
        )

    optimizer = BayesianOptimizer(objective_f=objective)
    optimizer.initialize(args.init_steps)

    for i in tqdm(range(args.optim_steps), desc="Optimization steps"):
        mean, std, ei = optimizer.step()
        fig = optimizer.visualize(mean, std, ei)

        if args.interactive:
            plt.show()

        if args.do_save:
            os.makedirs("./figures", exist_ok=True)
            fig.savefig(f"figures/fig_{i}.png")

    best_x, best_y = optimizer.get_current_max_point(mean)
    logging.info(
        "Optimization completed!\n"
        f"Estimated best learning rate is {best_x:.3f} with an accuracy of {best_y:.2f}."
    )


if __name__ == "__main__":
    main(_create_args())
    exit(0)

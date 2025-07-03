import torch
import os
import matplotlib.pyplot as plt

import argparse
from argparse import Namespace
from tqdm import tqdm

from src.dl.resnet import CustomLayerConfig
from src.bayesian_optim.bayesian_optimizer import BayesianOptimizer
from src.dl.trainer import FashionMNISTTrainer, Hyperparameters
from src.dl.resnet import CustomResNet


def _create_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="optimizer",
        description="Bayesian Optimizer for finding the best learning rate for a ResNet model on the Fashion MNIST dataset.",
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
        "--min-lr",
        type=float,
        default=1e-4,
        help="Minimum learning rate to check.",
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        default=1e-0,
        help="Maximum learning rate to check.",
    )
    parser.add_argument(
        "--no-log10",
        action="store_true",
        help="Uses linear scaling instead of logarithmic.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Uses a much less expensive objective function for debug purposes.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Flag for displaying the figure interactively for each iteration in a blocking manner.",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        default="./figures/",
        help="Where to save the figures, if specified.",
    )
    parser.add_argument(
        "--layers",
        nargs=6,
        type=int,
        default=[16, 16, 16, 16, 16, 16],
        help="The dimensionality of each of the six layers of the custom ResNet model. Used to customize the size of the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="The batch size to use for training the ResNet model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="How many epochs to use for each training run of the ResNet model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Which device to run the training on, e.g. 'cuda' or 'cpu'. "
        "Option 'auto' chooses cuda if available, else CPU. With multiple GPUs, specify the id with 'cuda:id', e.g.: 'cuda:0'.",
    )

    return parser.parse_args()


def main(args: Namespace) -> None:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    layer_cnf = CustomLayerConfig(*args.layers)
    model = CustomResNet(channels_in=1, n_classes=10, layer_cfg=layer_cnf)
    ml_trainer = FashionMNISTTrainer(model, device=device)

    # Define the objective function, sets up the trainer to train and test given a learning rate
    def objective(
        lr: torch.Tensor,
    ):  # could potentially be changed to select which param to optimize for (e.g. other than LR)
        torch.manual_seed(0)  # for more stable BO estimate
        hyperparams = Hyperparameters(
            epochs=args.epochs,
            learning_rate=lr.item(),
            batch_size=args.batch_size,
        )
        # Train, test and return the accuracy of the ResNet model
        return (
            ml_trainer.train_and_test(hparams=hyperparams)
            .unsqueeze(0)
            .detach()
            .cpu()
            .double()
        )

    if args.debug:
        objective = lambda x: (torch.sin(3.14 * 1.6 * x) + 4).log()

    optimizer = BayesianOptimizer(
        objective_f=objective,
        min_bound=args.min_lr,
        max_bound=args.max_lr,
        use_log_scale=not args.no_log10,
        debug=args.debug,
    )
    optimizer.initialize(args.init_steps)

    for i in tqdm(range(args.optim_steps), desc="Optimization steps", leave=False):
        mean, std, ei = optimizer.step()
        fig = optimizer.visualize(mean, std, ei)

        if args.interactive:
            plt.show()

        if args.save_path is not None:
            fig_path = os.path.join(args.save_path, f"fig_{i}.png")
            os.makedirs(args.save_path, exist_ok=True)
            fig.savefig(fig_path)

    best_x, best_y = optimizer.get_current_max_point(mean)
    print(
        "Optimization completed!\n"
        f"Estimated best learning rate is {best_x:.3f} with an accuracy of {best_y*100:.2f} %."
    )


if __name__ == "__main__":
    main(_create_args())
    exit(0)

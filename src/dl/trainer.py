import torch
import torchvision

import torch.utils
import torch.utils.data

from torch import nn
from tqdm import tqdm
from dataclasses import dataclass

from src.dl.resnet import CustomResNet


@dataclass
class Hyperparameters:
    """
    A wrapper class for the hyperparameters.
    """
    def __init__(self, epochs, learning_rate, batch_size) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size


class Accuracy:
    """
    Handles accuracy calculation in a HuggingFace-esque metric handling manner.  
    """
    def __init__(self) -> None:
        self.correct: int = 0
        self.total: int = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the internal accuracy handling (num correct and num total) for the input batch samples.

        :param logits: the raw, batched input logits.
        :param targets: the batched input targets.
        """
        pred = logits.argmax(-1)
        self.correct += pred.eq(targets).sum().item()
        self.total += targets.shape[0]

    def aggregate(self) -> torch.Tensor:
        """
        Aggregates the internal accuracy handling into a single accuracy score.

        :returns: the overall accuracy as a tensor.
        """
        return torch.tensor(self.correct / self.total)


class FashionMNISTTrainer:
    """
    A trainer for training on the FashionMNIST dataset. 

    :param model: the model to train.
    :param device: which device to train on, e.g. CUDA.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.step = 0
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # We are only using FashionMNIST in this demo anyway, just load it
        self.train_ds = torchvision.datasets.FashionMNIST(
            root="data/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        )

        self.test_ds = torchvision.datasets.FashionMNIST(
            root="data/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        )

    def train(self, hparams: Hyperparameters) -> None:
        """
        Trains the internal model using the provided hyperparameters.

        :param hparams: the hyperparameters to train with.
        """
        torch.manual_seed(0)

        lr = hparams.learning_rate
        epochs = hparams.epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

        data_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=hparams.batch_size
        )

        self.model.to(self.device).train()
        for _ in tqdm(
            range(epochs), desc="Training", leave=False, position=1, unit="Epochs"
        ):
            for batch in tqdm(
                data_loader,
                desc="Training steps",
                leave=False,
                position=2,
                unit="Batches",
            ):
                self._training_step(batch)

    def test(self, hparams: Hyperparameters) -> torch.Tensor:
        """
        Tests the (trained) internal model using the provided hyperparameters and returns the performance as accuracy as a tensor.

        :param hparams: the hyperparameters to test with.
        :returns: the accuracy as a tensor.
        """

        data_loader = torch.utils.data.DataLoader(
            self.test_ds, batch_size=hparams.batch_size
        )

        accuracy = Accuracy()
        self.model.to(self.device).eval()
        with torch.inference_mode():
            for batch in tqdm(data_loader, desc="Testing", leave=False, position=1):
                inputs, targets = self._extract_batch(batch)
                logits = self.model(inputs)

                accuracy.update(logits, targets)

        return accuracy.aggregate()

    def train_and_test(self, hparams: Hyperparameters) -> torch.Tensor:
        """
        A convenience function for using with Bayesian Optimization. Resets the model parameters, \
        then trains and tests the model using the provided hyperparameters, and finally returns the accuracy as a tensor.

        :param hparams: the hyperparameters to use.
        :returns: the accuracy as a tensor.
        """
        # Reset params to avoid reinstantiation
        self.reset_params()

        self.train(hparams=hparams)
        return self.test(hparams=hparams)

    def reset_params(self) -> None:
        """
        Resets the internal model's parameters.
        """
        for module in self.model.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _training_step(self, batch: list[torch.tensor, torch.tensor]) -> None:
        """
        Performs one trainin step on the provided batch.

        :param batch: the batch to train on.
        """
        self.optimizer.zero_grad()
        inputs, targets = self._extract_batch(batch)

        logits = self.model(inputs)
        loss: torch.Tensor = self.criterion(logits, targets)
        loss.backward()

        self.optimizer.step()

    def _extract_batch(self, batch: list[torch.tensor, torch.tensor]) -> tuple[torch.tensor, torch.tensor]:
        """
        Extracts the batch and sends each item to the device.

        :param batch: the batch to extract.
        :returns: the processed batch as a tuple.
        """
        # "extract" makes more sense when the batch is a dict, though.
        return (v.to(self.device) for v in batch)


if __name__ == "__main__":

    model = CustomResNet()
    trainer = FashionMNISTTrainer(model)

    hparams = Hyperparameters(
        epochs=5,
        learning_rate=0.1,
        batch_size=128,
    )
    print(trainer.train_and_test(hparams))
    print(trainer.train_and_test(hparams))
    print(trainer.train_and_test(hparams))

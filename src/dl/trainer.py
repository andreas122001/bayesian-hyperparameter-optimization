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
    def __init__(self, epochs, learning_rate, batch_size) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size


class Accuracy:
    def __init__(self) -> None:
        self.correct: int = 0
        self.total: int = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        pred = logits.argmax(-1)
        self.correct += pred.eq(targets).sum().item()
        self.total += targets.shape[0]

    def aggregate(self) -> torch.Tensor:
        return torch.tensor(self.correct / self.total)


class ResNetTrainer:
    def __init__(self, model: CustomResNet):
        self.step = 0
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
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

    def train(self, hparams: Hyperparameters):
        torch.manual_seed(0)

        lr = hparams.learning_rate
        epochs = hparams.epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

        data_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=hparams.batch_size
        )

        self.model.to(self.device).train()
        for _ in tqdm(range(epochs), desc="Training"):
            for batch in tqdm(data_loader, desc="Batches", leave=False):
                self._training_step(batch)

    def test(self, hparams: Hyperparameters) -> torch.Tensor:

        data_loader = torch.utils.data.DataLoader(
            self.test_ds, batch_size=hparams.batch_size
        )

        accuracy = Accuracy()
        self.model.to(self.device).eval()
        with torch.inference_mode():
            for batch in tqdm(data_loader, desc="Testing"):
                inputs, targets = self._extract_batch(batch)
                logits = self.model(inputs)

                accuracy.update(logits, targets)

        return accuracy.aggregate()

    def train_and_test(self, hparams: Hyperparameters) -> torch.Tensor:
        self.model.reset()  # reset params to avoid reinstantiation
        self.train(hparams=hparams)

        return self.test(hparams=hparams)

    def _training_step(self, batch):
        self.optimizer.zero_grad()
        inputs, targets = self._extract_batch(batch)

        logits = self.model(inputs)
        loss: torch.Tensor = self.criterion(logits, targets)
        loss.backward()

        self.optimizer.step()

    def _extract_batch(self, batch):
        return [v.to(self.device) for v in batch]


if __name__ == "__main__":

    model = CustomResNet()
    trainer = ResNetTrainer(model)

    hparams = Hyperparameters(
        epochs=5,
        learning_rate=0.1,
        batch_size=128,
    )
    print(trainer.train_and_test(hparams))
    print(trainer.train_and_test(hparams))
    print(trainer.train_and_test(hparams))

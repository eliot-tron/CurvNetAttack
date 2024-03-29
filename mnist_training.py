import argparse
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_networks import medium_cnn


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: Optimizer, epoch: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_interval = len(loader) // 10
    device = next(model.parameters()).device
    model.train()
    steps = []
    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if batch_idx % max(log_interval, 1) == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            steps.append(batch_idx)
        optimizer.step()
    steps = torch.tensor(steps)
    return steps


def test(model: nn.Module, loader: DataLoader) -> float:
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(loader.dataset),
            100.0 * correct / len(loader.dataset),
        )
    )
    return test_loss


def mnist_loader(batch_size: int, train: bool) -> DataLoader:
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data",
            train=train,
            download=False,
            transform=transforms.Compose(
                    [transforms.ToTensor(),
                     # transforms.Normalize((0.1307,), (0.3081,))
                     ]
            ),
        ),
        batch_size=batch_size,
        shuffle=train,
        num_workers=1,
        pin_memory=True,
    )
    return loader

def emnist_loader(batch_size: int, train: bool) -> DataLoader:
    dataset = datasets.EMNIST(
                "data_augmented",
                train=train,
                download=False,
                split='letters',
                transform=transforms.Compose(
                    [transforms.ToTensor(),
                     # transforms.Normalize((0.1307,), (0.3081,))
                     ]
                ),
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=1,
        pin_memory=True,
    )
    idx = torch.randint(dataset.data.shape[0]-1, (16,))
    idx = torch.arange(0, 31 ,3) + 31*10
    # save_images(dataset.data[idx,...], f'./data_augmented/visualisation_{"train" if train else "test"}', predictions=[dataset.classes[i] for i in dataset.targets[idx,...].int()])

    return loader


def exemplar_batch(batch_size: int, train: bool) -> torch.Tensor:
    dataset = datasets.MNIST(
        "data",
        train=train,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor()
             #, transforms.Normalize((0.1307,), (0.3081,))
             ]
        ),
    )
    examples = []
    for i in range(batch_size):
        examples.append(dataset[i][0])
    batch = torch.stack(examples, dim=0)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a basic model on MNIST",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoint",
        help="Model checkpoint output directory",
    )
    parser.add_argument(
        "--nl",
        type=str,
        metavar='f',
        default="ReLU",
        choices=['Sigmoid', 'ReLU'],
        help="Non linearity used by the network."
    )

    args = parser.parse_args(sys.argv[1:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    non_linearity = args.nl
    non_linearity_dict = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
    }

    output_dir = Path(args.output_dir).expanduser() / "MNIST" / f"lr={args.lr}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # model = medium_cnn(num_classes=27)  # 26 letters and 1 N/A
    model = medium_cnn(num_classes=10, non_linearity=non_linearity_dict[non_linearity])
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader = mnist_loader(args.batch_size, train=True)
    test_loader = mnist_loader(args.batch_size, train=False)

    global_steps = []
    global_ranks = []
    global_traces = []
    for epoch in range(args.epochs):
        epoch_steps = train_epoch(
            model, train_loader, optimizer, epoch + 1
        )
        global_steps.append(epoch_steps + epoch * len(train_loader))
        test(model, test_loader)
        torch.save(model.state_dict(), output_dir / f"medium_cnn_{epoch + 1:02d}.pt")

    global_steps = torch.cat(global_steps, dim=0)
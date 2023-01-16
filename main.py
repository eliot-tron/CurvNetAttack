from os import makedirs, path
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from adversarial_attack import (OneStepSpectralAttack,
                                StandardTwoStepSpectralAttack)
from mnist_networks import medium_cnn
from xor_networks import xor_net
from xor_datasets import XorDataset
from foliation import Foliation
from torch import nn

if __name__ == "__main__":
    dataset_name = ["MNIST", "XOR"][0]
    num_samples = int(200)
    task = ["", "plot-attack", "plot-attacks-2D", "fooling-rates", "plot-leaves"][3]

    if dataset_name == "MNIST":
        checkpoint_path = './checkpoint/medium_cnn_10.pt'
        network = medium_cnn(checkpoint_path, non_linearity=nn.Sigmoid())
        network_score = medium_cnn(checkpoint_path, score=True)

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        input_space = datasets.MNIST(
            "data",
            train=False,  # TODO: True ?
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
    elif dataset_name == "XOR":
        checkpoint_path = './checkpoint/xor_net_10.pt'
        # checkpoint_path = './checkpoint/xor_net_sigmoid_22.pt'
        network = xor_net(checkpoint_path)
        network_score = xor_net(checkpoint_path, score=True)

        input_space = XorDataset(
            nsample=10000,
            discrete=False,
        )

    STSSA = StandardTwoStepSpectralAttack(
                network=network,
                network_score=network_score,
            )
    OSSA = OneStepSpectralAttack(
                network=network,
                network_score=network_score,
            )
    
    foliation = Foliation(
        network=network,
        network_score=network_score,
    )
    
    print(f'Task {task} with dataset {dataset_name} and {num_samples} samples.')

    if task in ["", "plot-attack", "fooling-rates", "plot-attacks-2D"]:
        if task == "plot-attack":
            num_samples = 1
        
        if num_samples > len(input_space):
            print(f'WARNING: you are trying to get mmore samples ({num_samples}) than the number of data in the test set ({len(input_space)})')
        random_indices = torch.randperm(len(input_space))[:num_samples]
        input_points = torch.stack([input_space[idx][0] for idx in random_indices])

    if task == "plot-attack":
        two_step_attack = STSSA.compute_attack(input_points, budget=1)
        plt.matshow(input_points[0][0] - two_step_attack.detach().numpy()[0][0])
        plt.show()
        one_step_attack = OSSA.compute_attack(
                                input_points,
                                budget=1
                            )

        plt.matshow(input_points[0][0] - one_step_attack.detach().numpy()[0][0])
        plt.show()

        plt.matshow(two_step_attack.detach().numpy()[0][0] - one_step_attack.detach().numpy()[0][0])
        plt.show()
    
    if task == "fooling-rates":
        MAX_BUDGET = 10
        STSSA.save_fooling_rates(input_points, step=0.1, end=MAX_BUDGET)
        OSSA.save_fooling_rates(input_points, step=0.1, end=MAX_BUDGET)
        plt.legend()
        savedirectory = f"./output/{dataset_name}/"
        if not path.isdir(savedirectory):
            makedirs(savedirectory)
        savename = f"fooling_rates_compared_nsample={num_samples}"
        savepath = savedirectory + ("" if savedirectory[-1] == "/" else "/") + savename
        plt.savefig(savepath + '.pdf')
        plt.clf()

    if task == "plot-attacks-2D":
        foliation.plot(eigenvectors=False)
        STSSA.plot_attacks_2D(test_points=input_points,budget=0.1)
        foliation.plot(eigenvectors=False)
        OSSA.plot_attacks_2D(test_points=input_points, budget=0.1)
    
    if task == "plot-leaves":
        foliation.plot(eigenvectors=False)
    
    # num_samples = 1
    # STSSA.test_jac_proba(input_points)
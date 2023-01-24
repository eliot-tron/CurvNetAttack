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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(f"Device: {device}")
    dataset_name = ["MNIST", "XOR"][0]
    num_samples = 1000
    task = ["", "plot-attack", "plot-attacks-2D", "fooling-rates", "plot-leaves"][3]
    non_linearity = ['Sigmoid', 'ReLU'][0]
    if dataset_name == "MNIST":
        MAX_BUDGET = 10
        STEP_BUDGET = 0.1
    elif dataset_name == "XOR":
        MAX_BUDGET = 1
        STEP_BUDGET = 0.01

    if dataset_name == "MNIST":
        if non_linearity == 'Sigmoid':
            checkpoint_path = './checkpoint/medium_cnn_10_Sigmoid.pt'
            non_linearity = nn.Sigmoid()
        elif non_linearity == 'ReLU':
            checkpoint_path = './checkpoint/medium_cnn_10_ReLU.pt'
            non_linearity = nn.ReLU()
        network = medium_cnn(checkpoint_path, non_linearity=non_linearity)
        network_score = medium_cnn(checkpoint_path, score=True, non_linearity=non_linearity)

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        input_space = datasets.MNIST(
            "data",
            train=False,  # TODO: True ?
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
    elif dataset_name == "XOR":
        if non_linearity == 'Sigmoid':
            checkpoint_path = './checkpoint/xor_net_22_Sigmoid.pt'
            print("/!\\ "*5 + "\nSigmoid for Xor not working.")
            non_linearity = nn.Sigmoid()
        elif non_linearity == 'ReLU':
            checkpoint_path = './checkpoint/xor_net_30_ReLU.pt'
            non_linearity = nn.ReLU()
        print("Loading net")
        network = xor_net(checkpoint_path, non_linearity=non_linearity)
        network_score = xor_net(checkpoint_path, score=True, non_linearity=non_linearity)
        print("Networ loaded")

        input_space = XorDataset(
            nsample=10000,
            discrete=False,
        )
        print("dataset loaded")

    network = network.to(device)
    network_score = network_score.to(device)

    print("network to cuda done")

    STSSA = StandardTwoStepSpectralAttack(
                network=network,
                network_score=network_score,
            )
    OSSA = OneStepSpectralAttack(
                network=network,
                network_score=network_score,
            )
        
    print("attacks created")
    
    foliation = Foliation(
        network=network,
        network_score=network_score,
    )

    print("foliation created")
    
    print(f'Task {task} with dataset {dataset_name} and {num_samples} samples.')

    if task in ["", "plot-attack", "fooling-rates", "plot-attacks-2D"]:
        if task == "plot-attack":
            num_samples = 1
        
        if num_samples > len(input_space):
            print(f'WARNING: you are trying to get mmore samples ({num_samples}) than the number of data in the test set ({len(input_space)})')
        random_indices = torch.randperm(len(input_space))[:num_samples]
        input_points = torch.stack([input_space[idx][0] for idx in random_indices])
        input_points = input_points.to(device)

    if task == "plot-attack":
        plt.matshow(input_points[0][0])
        plt.show()
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
        savedirectory = f"./output/{dataset_name}/"
        if not path.isdir(savedirectory):
            makedirs(savedirectory)
        STSSA.save_fooling_rates(input_points, step=STEP_BUDGET, end=MAX_BUDGET, savepath=savedirectory + f"fooling_rates_nsample={num_samples}")
        OSSA.save_fooling_rates(input_points, step=STEP_BUDGET, end=MAX_BUDGET, savepath=savedirectory + f"fooling_rates_nsample={num_samples}")
        plt.legend()
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
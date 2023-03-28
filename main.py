import argparse
from os import makedirs, path
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from adversarial_attack import (OneStepSpectralAttack,
                                StandardTwoStepSpectralAttack)
from adversarial_attack_plots import compare_fooling_rates, compare_inf_norm, plot_attacks_2D, plot_curvature_2D
from mnist_networks import medium_cnn
from xor_networks import xor_net
from xor_datasets import XorDataset
from foliation import Foliation
from torch import nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute a One-Step or Two-Step spectral attack, and some visualizations.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=['MNIST', 'XOR'],
        help="Dataset name to be used.",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        metavar='N',
        default=128,
        help="Number of points to compute the attack on."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="fooling-rates",
        choices=['plot-attack', 'plot-attacks-2D', 'fooling-rates', 'plot-leaves', 'plot-curvature', 'inf-norm'],
        help="Task."
    )
    parser.add_argument(
        "--nl",
        type=str,
        metavar='f',
        default="relu",
        choices=['Sigmoid', 'ReLU'],
        help="Non linearity used by the network."
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(f"Device: {device}")
    
    dataset_name = args.dataset
    num_samples = args.nsample
    task = args.task
    non_linearity = args.nl

    if dataset_name == "MNIST":
        MAX_BUDGET = 10
        STEP_BUDGET = 1
    elif dataset_name == "XOR":
        MAX_BUDGET = 1
        STEP_BUDGET = 0.01
    
    if task == "plot-curvature":
        dataset_name = "XOR"

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
            checkpoint_path = './checkpoint/xor_net_sigmoid_20.pt'
            # print("/!\\ "*5 + "\nSigmoid for Xor not working.")
            non_linearity = nn.Sigmoid()
        elif non_linearity == 'ReLU':
            checkpoint_path = './checkpoint/xor_net_relu_30.pt'
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

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        network = nn.DataParallel(network)
        network_score = nn.DataParallel(network_score)

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

    if task in ["", "plot-attack", "fooling-rates", "plot-attacks-2D", "inf-norm"]:
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
        savename = f"fooling_rates_compared_nsample={num_samples}"
        savepath = savedirectory + ("" if savedirectory[-1] == "/" else "/") + savename
        compare_fooling_rates(
            [STSSA, OSSA],
            input_points,
            step=STEP_BUDGET,
            end=MAX_BUDGET,
            savepath=savepath
        )

    if task == "inf-norm":
        savedirectory = f"./output/{dataset_name}/"
        if not path.isdir(savedirectory):
            makedirs(savedirectory)
        savename = f"inf_norm_compared_nsample={num_samples}"
        savepath = savedirectory + ("" if savedirectory[-1] == "/" else "/") + savename
        compare_inf_norm(
            [STSSA, OSSA],
            input_points,
            step=STEP_BUDGET,
            end=MAX_BUDGET,
            savepath=savepath
        )

    if task == "plot-attacks-2D":
        foliation.plot(eigenvectors=False)
        plot_attacks_2D(STSSA, test_points=input_points,budget=0.1)
        foliation.plot(eigenvectors=False)
        plot_attacks_2D(OSSA, test_points=input_points, budget=0.1)
    
    if task == "plot-leaves":
        foliation.plot(eigenvectors=False)
    
    if task == "plot-curvature":
        plot_curvature_2D(STSSA)
        foliation.plot(eigenvectors=False, transverse=True)
        plt.show()
    
    # num_samples = 1
    # STSSA.test_jac_proba(input_points)
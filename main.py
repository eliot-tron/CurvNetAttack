import argparse
from ctypes import ArgumentError
from os import makedirs, path
import random
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from adversarial_attack import (APGDAttack, AdversarialAutoAttack, OneStepSpectralAttack,
                                TwoStepSpectralAttack)
from adversarial_attack_plots import compare_fooling_rates, compare_inf_norm, plot_attacks_2D, plot_contour_2D, plot_curvature_2D
from mnist_networks import medium_cnn
from xor_networks import xor_net, xor_net_old
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
        choices=['MNIST', 'XOR', 'XOR-old'],
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
        choices=['plot-attack', 'plot-attacks-2D', 'fooling-rates', 'plot-leaves', 'plot-curvature', 'plot-contour','inf-norm', 'save-attacks'],
        help="Task."
    )
    parser.add_argument(
        "--nl",
        type=str,
        metavar='f',
        default="ReLU",
        choices=['Sigmoid', 'ReLU'],
        help="Non linearity used by the network."
    )
    parser.add_argument(
        "--startidx",
        type=int,
        default=0,
        help="Start index of the input points"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Permutes randomly the inputs."
    )
    parser.add_argument(
        "--run",
        type=str,
        metavar="attack",
        nargs="+",
        default=["TSSA", "OSSA"],
        choices=["TSSA", "OSSA", "AA", "APGD"],
        help="List of attacks to run on the task."
    )
    parser.add_argument(
        "--attacks",
        type=str,
        metavar="path",
        nargs="+",
        default=None,
        help="Path to (budget, test_point, attack_vectors) if you had them precomputed."
    )
    parser.add_argument(
        "--range",
        type=float,
        metavar=("a","b"),
        nargs=2,
        default=[0.,1.],
        help="Range for the generated datapoints for the Xor task. Datapoints will lie in [a,b)²."
    )
    parser.add_argument(
        "--savedirectory",
        type=str,
        metavar='path',
        default='./output/',
        help="Path to the directory to save the outputs in."
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(f"Device: {device}")
    
    dataset_name = args.dataset
    num_samples = args.nsample
    task = args.task
    non_linearity = args.nl
    start_index = args.startidx
    attacks_to_run_str = args.run
    attack_paths = args.attacks
    xor_data_range = args.range

    batch_size = 125
    savedirectory = args.savedirectory + ("" if args.savedirectory[-1] == '/' else '/') + f"{dataset_name}/{task}/"
    if not path.isdir(savedirectory):
        makedirs(savedirectory)

    if not args.random:
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        

    if dataset_name == "MNIST":
        MAX_BUDGET = 6
        STEP_BUDGET = 0.1
    elif dataset_name[:3] == "XOR":
        MAX_BUDGET = 0.5
        STEP_BUDGET = 0.005
    
    if task in ["plot-curvature", "plot-contour"] and dataset_name[:3] != 'XOR':
        raise ValueError(f'Plot curvature requires a 2D dataset and not {dataset_name}.')

    if dataset_name == "MNIST":
        if non_linearity == 'Sigmoid':
            checkpoint_path = './checkpoint/medium_cnn_30_Sigmoid.pt'
            non_linearity = nn.Sigmoid()
        elif non_linearity == 'ReLU':
            checkpoint_path = './checkpoint/medium_cnn_10_ReLU.pt'
            non_linearity = nn.ReLU()
        network = medium_cnn(checkpoint_path, non_linearity=non_linearity)
        network_score = medium_cnn(checkpoint_path, score=True, non_linearity=non_linearity)

        # normalize = transforms.Normalize((0.1307,), (0.3081,))

        input_space = datasets.MNIST(
            "data",
            train=False,  # TODO: True ?
            download=True,
            transform=transforms.Compose([transforms.ToTensor()
                                          # , normalize
                                        ]),
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
        print("Network loaded")

        input_space = XorDataset(
            nsample=10000,
            discrete=False,
            range=xor_data_range,
        )
        print("dataset loaded")
    elif dataset_name == "XOR-old":
        if non_linearity == 'Sigmoid':
            checkpoint_path = './checkpoint/old/xor_net_sigmoid_20.pt'
            # print("/!\\ "*5 + "\nSigmoid for Xor not working.")
            non_linearity = nn.Sigmoid()
        elif non_linearity == 'ReLU':
            checkpoint_path = './checkpoint/old/xor_net_relu_10.pt'
            non_linearity = nn.ReLU()
        print("Loading net")
        network = xor_net_old(checkpoint_path, non_linearity=non_linearity)
        network_score = xor_net_old(checkpoint_path, score=True, non_linearity=non_linearity)
        print("Network loaded")

        input_space = XorDataset(
            nsample=10000,
            discrete=False,
            range=xor_data_range,
        )
        print("dataset loaded")

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        network = nn.DataParallel(network)
        network_score = nn.DataParallel(network_score)

    network = network.to(device)
    network_score = network_score.to(device)

    print(f"network to {device} done")

    attacks_to_run = []

    for attack_name in attacks_to_run_str:
        if attack_name.casefold() == 'tssa':
            TSSA = TwoStepSpectralAttack(
                    network=network,
                    network_score=network_score,
                )
            attacks_to_run.append(TSSA)
        if attack_name.casefold() == 'ossa':
            OSSA = OneStepSpectralAttack(
                        network=network,
                        network_score=network_score,
                    )
            attacks_to_run.append(OSSA)
        if attack_name.casefold() == 'aa':
            AA = AdversarialAutoAttack(
                        network=network,
                        network_score=network_score,
            )
            attacks_to_run.append(AA)
        if attack_name.casefold() == 'apgd':
            APGD = APGDAttack(
                        network=network,
                        network_score=network_score,
            )
            attacks_to_run.append(APGD)
        
    print(f"attacks created: {[a.__name__ for a in attacks_to_run]}")
    
    foliation = Foliation(
        network=network,
        network_score=network_score,
    )

    print("foliation created")
    
    
    if attack_paths is not None:
        budget_range_list, input_points_list, attack_vectors = [], [], []
        print("Loading precomputed attacks", end='')
        for attack_path in attack_paths:
            print(".", end='')
            br, ip, av = torch.load(attack_path, map_location=device)
            budget_range_list.append(br)
            input_points_list.append(ip)
            attack_vectors.append(av)
        budget_range = budget_range_list[0]
        input_points = input_points_list[0]
        print("done")
        for br in budget_range_list:
            if not torch.allclose(budget_range, br):
                raise ValueError("The loaded attacks must have the same budgets.")
        for ip in input_points_list:
            if not torch.allclose(input_points, ip):
                raise ValueError("The loaded attacks must have the same input points.")
        num_samples = input_points.shape[0]
    else:
        budget_range = (MAX_BUDGET, STEP_BUDGET)
        attack_vectors = None

    print(f'Task {task} with dataset {dataset_name} and {num_samples} samples.')

    # Initialization of the input points
    if task in ["", "plot-attack", "fooling-rates", "plot-attacks-2D", "inf-norm", "save-attacks"] and attack_paths is None:
        if task == "plot-attack":
            num_samples = 1
        
        if num_samples > len(input_space):
            print(f'WARNING: you are trying to get more samples ({num_samples}) than the number of data in the test set ({len(input_space)})')
        
        if args.random:
            indices = torch.randperm(len(input_space))[:num_samples]
        else:
            indices = range(start_index, start_index + num_samples)

        input_points = torch.stack([input_space[idx][0] for idx in indices])
        input_points = input_points.to(device)


    if task == "plot-attack":
        plt.matshow(input_points[0][0])
        plt.show()
        for adversarial_attack in attacks_to_run:
            attack_output = adversarial_attack.compute_attack(input_points, budget=1)
            plt.matshow(input_points[0][0] - attack_output.detach().numpy()[0][0])
            plt.show()
    
    if task == "save-attacks":
        savename = f"attacks_nsample={num_samples}_start={start_index}"
        savepath = savedirectory + ("" if savedirectory[-1] == "/" else "/") + savename

        if device.type == 'cuda':
            batched_input_points = torch.split(input_points, batch_size , dim=0)
        else: # enough memory in cpu for a single batch
            batched_input_points = [input_points]

        for batch_index, batch in enumerate(batched_input_points):
            print(f"Batch number {batch_index} starting...")
            for adversarial_attack in attacks_to_run:
                adversarial_attack.save_attack(
                    test_points=batch,
                    budget_step=STEP_BUDGET,
                    budget_max=MAX_BUDGET,
                    savepath=savepath + f"_batch={batch_index}"
                )
            torch.cuda.empty_cache()
    
    if task == "fooling-rates":
        savedirectory += ("" if savedirectory[-1] == "/" else "/") + '-'.join(sorted(attacks_to_run_str)).casefold()
        savename = f"fooling_rates_compared_nsample={num_samples}_start={start_index}_nl={non_linearity}"
        savepath = savedirectory + ("" if savedirectory[-1] == "/" else "/") + savename

        if device.type == 'cuda' and attack_paths is None:
            batched_input_points = torch.split(input_points, batch_size , dim=0)
        else: # enough memory in cpu for a single batch
            batched_input_points = [input_points]

        for batch_index, batch in enumerate(batched_input_points):
            print(f"Batch number {batch_index} starting...")
            compare_fooling_rates(
                attacks_to_run,
                batch,
                budget_range=budget_range,
                savepath=savepath + f"_batch={batch_index}" + (f"I={xor_data_range}" if dataset_name == 'XOR' else ""),
                attack_vectors=attack_vectors
            )

    if task == "inf-norm":
        savedirectory += ("" if savedirectory[-1] == "/" else "/") + '-'.join(sorted(attacks_to_run_str)).casefold()
        savename = f"inf_norm_compared_nsample={num_samples}"
        savepath = savedirectory + ("" if savedirectory[-1] == "/" else "/") + savename

        if device.type == 'cuda':
            batched_input_points = torch.split(input_points, batch_size , dim=0)
        else: # enough memory in cpu for a single batch
            batched_input_points = [input_points]

        for batch_index, batch in enumerate(batched_input_points):
            print(f"Batch number {batch_index} starting...")
            compare_inf_norm(
                attacks_to_run,
                batch,
                budget_range=budget_range,
                savepath=savepath,
                attack_vectors=attack_vectors
            )

    if task == "plot-attacks-2D":
        colors_dict = {'tssa': 'blue',
                       'ossa': 'orange',
                       'aa': 'red',
                       'apdg': 'green'}
        foliation.plot(eigenvectors=False)
        for name, adversarial_attack in zip(attacks_to_run_str, attacks_to_run):
            plot_attacks_2D(adversarial_attack, test_points=input_points,budget=0.1, color=colors_dict[name.casefold()])
            # foliation.plot(eigenvectors=False)
            # plt.legend()
        savedirectory += ("" if savedirectory[-1] == "/" else "/") + '-'.join(sorted(attacks_to_run_str)).casefold()
        savename = f"plot_attacks_2D_budget=1e-1_nsample={num_samples}_nl={non_linearity}"
        savepath = savedirectory + ("" if savedirectory[-1] == "/" else "/") + savename
        plt.savefig(savepath + '.pdf', format='pdf')
    
    if task == "plot-leaves":
        foliation.plot(eigenvectors=False)
    
    if task == "plot-curvature":
        plot_curvature_2D(attacks_to_run[0])
        foliation.plot(eigenvectors=False, transverse=True)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        savename = f"plot_curvature_nl={non_linearity}"
        savepath = savedirectory + ("" if savedirectory[-1] == "/" else "/") + savename
        plt.savefig(savepath + f'.pdf', dpi=None, transparent=True)
        # plt.show()
    
    if task == "plot-contour":
        plot_contour_2D(attacks_to_run[0])
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        savename = f"plot_curvature_nl={non_linearity}"
        savepath = savedirectory + ("" if savedirectory[-1] == "/" else "/") + savename
        plt.savefig(savepath + f'.pdf', dpi=None, transparent=True)
        plt.show()
    
    # num_samples = 1
    # TSSA.test_jac_proba(input_points)
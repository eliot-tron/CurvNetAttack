import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from adversarial_attack import (OneStepSpectralAttack,
                                StandardTwoStepSpectralAttack)
from mnist_networks import medium_cnn

if __name__ == "__main__":
    MNIST = True
    num_samples = int(1e1)
    if MNIST:
        checkpoint_path = './checkpoint/medium_cnn_10.pt'
        network = medium_cnn(checkpoint_path)
        network_score = medium_cnn(checkpoint_path, score=True)

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        input_space = datasets.MNIST(
            "data",
            train=False,  # TODO: True ?
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

    STSSA = StandardTwoStepSpectralAttack(
                network=network,
                network_score=network_score,
            )
    if num_samples > len(input_space):
        print(f'WARNING: you are trying to get mmore samples ({num_samples}) than the number of data in the test set ({len(input_space)})')
    random_indices = torch.randperm(len(input_space))[:num_samples]
    input_points = torch.stack([input_space[idx][0] for idx in random_indices])
    # two_step_attack = STSSA.compute_attack(input_points, budget=28)
    # plt.matshow(input_points[0][0] - two_step_attack.detach().numpy()[0][0])
    # plt.show()
    STSSA.save_fooling_rates(input_points, step=1, end=28)

    OSSA = OneStepSpectralAttack(
                network=network,
                network_score=network_score,
            )
    OSSA.save_fooling_rates(input_points, step=1, end=28)
    plt.legend()
    savepath = f"./output/fooling_rates_compared_nsample={num_samples}" + ("_MNIST" if MNIST else "")
    plt.savefig(savepath + '.pdf')
    plt.clf()
    # one_step_attack = OSSA.compute_attack(
    #                         input_points,
    #                         budget=1e-3
    #                     )

    # plt.matshow(input_points[0][0] - two_step_attack.detach().numpy()[0][0])
    # plt.show()

    # plt.matshow(two_step_attack.detach().numpy()[0][0] - one_step_attack.detach().numpy()[0][0])
    # plt.show()
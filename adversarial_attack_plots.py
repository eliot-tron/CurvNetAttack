"""Module implementing various plot functions for comparing attacks."""
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from geometry import GeometricModel
from adversarial_attack import AdversarialAttack, StandardTwoStepSpectralAttack, OneStepSpectralAttack
from matplotlib import cm
from matplotlib.colors import SymLogNorm

def compare_fooling_rates(
    adversarial_attacks: list[AdversarialAttack],
    test_points: torch.tensor,
    step: float=1e-2,
    end: float=1.,
    savepath: str="./output/fooling_rates_compared"
    ) -> None:
    """Saves the graph of fooling rates with respect to the budget.
    :adversarial_attacks: list of attacks to compare.
    :test_point: points to compute the fooling rates on.
    :step: step size between two budgets.
    :end: max budget.
    :returns: None
    """
    
    for attack in adversarial_attacks:
        budget_range = torch.arange(0, end, step).cpu()
        fooling_rates = [attack.test_attack(budget, test_points).cpu() for budget in tqdm(budget_range)]
        plt.plot(budget_range, fooling_rates, label=type(attack).__name__)
        torch.save((budget_range, fooling_rates), savepath + f"_{type(attack).__name__}_budget-rates.pt")

    plt.xlabel("Budget")
    plt.ylabel("Fooling rate")
    plt.legend()
    plt.savefig(savepath + '.pdf', format='pdf')
    plt.clf()

def compare_inf_norm(
    adversarial_attacks: list[AdversarialAttack],
    test_points: torch.tensor,
    step: float=1e-2,
    end: float=1.,
    savepath: str="./output/infinity_norm_compared"
    ) -> None:
    """Save the plot of the infinity norm with respect to the euclidean budget.

    Args:
        adversarial_attacks: list of attacks to compare.
        test_point: points to compute the fooling rates on.
        step (float, optional): stem size between two budgets. Defaults to 1e-2.
        end (int, optional): max budget. Defaults to 1.
        savepath (str, optional): Path to save the plot to. Defaults to "./output/infty_norm".

    Returns:
        _type_: None
    """

    budget_range = torch.arange(0, end, step).cpu()
    attack_vectors = []
    
    for attack in adversarial_attacks:
        attack_vectors.append([attack.compute_attack(test_points, budget) - test_points for budget in tqdm(budget_range)])
    
    for attack, attack_vec in zip(adversarial_attacks, attack_vectors):
        infty_norms = torch.linalg.vector_norm(attack_vec[-1].reshape(attack_vec[-1].shape[0], -1), ord=float('inf'), dim=1)
        imax = infty_norms.argmax(dim=0)

        figure, axes = plt.subplots(1, 3)
        im = axes[0].matshow(test_points[imax].squeeze(0).detach())
        axes[0].set_title("Test point")
        axes[1].matshow(attack_vec[-1][imax].squeeze(0).detach())
        axes[1].set_title("Attack vector")
        axes[2].matshow((attack_vec[-1][imax] + test_points[imax]).squeeze(0).detach())
        axes[2].set_title("Attacked point")
        figure.colorbar(im, ax=axes.ravel().tolist())
        [ax.set_axis_off() for ax in axes.ravel()]
        plt.savefig(savepath + f"_{type(attack).__name__}_worst-case.pdf")
        # plt.show()
        figure.clf()

    for attack, attack_vec in zip(adversarial_attacks, attack_vectors):
        inf_norm = [torch.linalg.vector_norm(av, ord=float('inf'), dim=1).max().cpu().detach() for av in attack_vec]
        plt.plot(budget_range, inf_norm, label=type(attack).__name__)
        torch.save((budget_range, inf_norm), savepath + '_budget-infnorm.pt')

    plt.xlabel("Budget")
    plt.ylabel("Infinity norm")
    plt.legend()
    plt.savefig(savepath + '.pdf', format='pdf')
    # plt.show()
    plt.clf()

def test_points_2D(nb_points, size=1):
    """Generate test points uniformly in the square [0.5-size/2, 0.5+size/2]^2."""
    return (torch.rand(nb_points, 2) - 0.5)* size + 0.5

def plot_attacks_2D(adversarial_attack, test_points, budget=0.3):
    """Plots the attack vectors on the input space."""        
    adversarial_attack.task = "xor"
    attack_vectors = adversarial_attack.compute_attack(test_points, budget) - test_points
    attack_vectors = attack_vectors.detach()
    for coords, attack_vector in tqdm(zip(test_points, attack_vectors)):
        plt.quiver(coords[0], coords[1], attack_vector[0], attack_vector[1], width=0.003, scale_units='xy', angles='xy', scale=1, zorder=2)
    if adversarial_attack.task == "xor":
        plt.plot([0, 1], [0, 1], "ro", zorder=3)
        plt.plot([0, 1], [1, 0], "go", zorder=3)
    elif adversarial_attack.task == "or":
        plt.plot([0], [0], "ro", zorder=3)
        plt.plot([0, 1, 1], [1, 0, 1], "go", zorder=3)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    savepath = "./output/attacks_svd_{}_{}".format(type(adversarial_attack).__name__, adversarial_attack.task)
    plt.savefig(savepath + '.pdf', format='pdf')
    plt.clf()

def plot_curvature_2D(adversarial_attack):
    """Plot the extrinsic curvature for a 2D input manifold and 1D leaves.
    """

    xs = torch.linspace(0, 1, steps=100)
    grid = torch.cartesian_prod(xs, xs)
    p = adversarial_attack.proba(grid)[..., 1]
    p = p.reshape((*xs.shape, *xs.shape))
    plt.pcolormesh(xs, xs, p.detach().numpy()) 
    # levels = torch.logspace(-16, 0, 20)
    # plt.contour(xs, xs, p.detach().numpy(), levels=levels)
    plt.colorbar()
    plt.show()

    G_1 = adversarial_attack.local_data_matrix(grid)
    if G_1.is_cuda:
        _, _, v_1 = torch.linalg.svd(G_1)
        normal = v_1[..., 0]  # be careful, it isn't intuitive -> RTD
    else:
        _, v_1 = torch.linalg.eigh(G_1)  # value, vector, in ascending order
        normal = v_1[..., -1]  # be careful, it isn't intuitive -> RTD
    norm_1 = torch.linalg.vector_norm(normal, ord=2, dim=-1, keepdim=True)
    normal = normal / norm_1
    normal = normal.reshape(grid.shape)

    """Computing first step's sign."""
    # print("compute attack sign")
    normal_sign = adversarial_attack.attack_sign(grid, normal)
    normal = torch.einsum('z, z... -> z...', normal_sign, normal)

    dx = 1e-3 * normal
    print(f"grid: {grid.shape} and dx: {dx.shape}")
    print(f"grid + dx: {(grid+dx).shape}")
    G_dx = adversarial_attack.local_data_matrix(grid + dx)

    if G_dx.is_cuda:
        _, _, v_dx = torch.linalg.svd(G_dx)
        normal_dx = v_dx[..., 0]  # be careful, it isn't intuitive -> RTD
    else:
        _, v_dx = torch.linalg.eigh(G_dx)  # value, vector, in ascending order
        normal_dx = v_dx[..., -1]  # be careful, it isn't intuitive -> RTD
    
    """Computing second step's sign."""
    normal_dx_sign = torch.einsum('z..., z... -> z', normal, normal_dx).sign()
    normal_dx_sign[normal_dx_sign==0] = 1
    normal_dx = torch.einsum('z, z... -> z...', normal_dx_sign, normal_dx)


    cross = normal[..., 0] * normal_dx[..., 1] - normal[..., 1] * normal_dx[..., 0]
    print(f"cross: {cross.shape}")

    dtheta = torch.asin(cross)
    
    """3D plot"""
    # X, Y = torch.meshgrid(xs, xs)
    # dtheta = dtheta.reshape(X.shape)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(X, Y, dtheta.detach(), cmap=cm.Blues)

    """2D color plot"""
    dtheta = dtheta.reshape((*xs.shape, *xs.shape)).detach()
    plt.pcolormesh(xs, xs, dtheta, cmap='PRGn')
    plt.colorbar()

    plt.show()
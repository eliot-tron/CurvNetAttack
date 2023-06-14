"""Module implementing various plot functions for comparing attacks."""
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from adversarial_attack import AdversarialAttack, TwoStepSpectralAttack, OneStepSpectralAttack
from typing import Tuple, Union
from matplotlib import cm
from matplotlib.colors import SymLogNorm

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def compare_fooling_rates(
    adversarial_attacks: list[AdversarialAttack],
    test_points: torch.Tensor,
    budget_range: Union[torch.Tensor, Tuple[float, float]]=(1., 1e-2),
    savepath: str="./output/fooling_rates_compared",
    attack_vectors: list[list[torch.Tensor]]=None,
    ) -> None:
    """Saves the graph of fooling rates with respect to the budget.
    :adversarial_attacks: list of attacks to compare.
    :test_point: points to compute the fooling rates on.
    :budget_range: tensor of budgets or tuple (max_budget, step_size) to generate an arange from 0.
    :attack_vectors: precomputed attack vectors for each adversarial_attack classes.
    :returns: None
    """
    if not torch.is_tensor(budget_range):
        end, step = budget_range
        budget_range = torch.arange(0, end, step).cpu()

    if attack_vectors is None:
        for attack in adversarial_attacks:
            fooling_rates = [attack.test_attack(budget, test_points).cpu() for budget in tqdm(budget_range)]
            plt.plot(budget_range.cpu(), fooling_rates, label=type(attack).__name__)
            torch.save((budget_range, fooling_rates), savepath + f"_{type(attack).__name__}_budget-rates.pt")
    else:
        for attack, attack_by_budget in zip(adversarial_attacks, attack_vectors):
            fooling_rates = [attack.test_attack(budget, test_points, av).cpu() for budget, av in tqdm(zip(budget_range, attack_by_budget))]
            plt.plot(budget_range.cpu(), fooling_rates, label=type(attack).__name__)
            torch.save((budget_range, fooling_rates), savepath + f"_{type(attack).__name__}_budget-rates.pt")

    plt.xlabel("Budget")
    plt.ylabel("Fooling rate")
    plt.legend()
    plt.savefig(savepath + '.pdf', format='pdf')
    plt.clf()

def compare_inf_norm(
    adversarial_attacks: list[AdversarialAttack],
    test_points: torch.Tensor,
    budget_range: Union[torch.Tensor, Tuple[float, float]]=(1., 1e-2),
    savepath: str="./output/infinity_norm_compared",
    attack_vectors: list[list[torch.Tensor]]=None,
    plot_inf_chart: bool=False,
    restrict_to_class: int=None,
    ) -> None:
    """Save the plot of the infinity norm with respect to the euclidean budget.

    Args:
        adversarial_attacks: list of attacks to compare.
        test_point: points to compute the fooling rates on.
        step (float, optional): stem size between two budgets. Defaults to 1e-2.
        end (int, optional): max budget. Defaults to 1.
        savepath (str, optional): Path to save the plot to. Defaults to "./output/infty_norm".
        attack_vectors: precomputed attack vetors for each adversarial attack classes.
        plot_inf_chart: Plot also the chart of the infinity norm wrt the budget.
        restrict_to_class: Only choose amoung the class [restrict_to_class].
    Returns:
        _type_: None
    """

    if not torch.is_tensor(budget_range):
        end, step = budget_range
        budget_range = torch.arange(0, end, step).cpu()
    
    if restrict_to_class is not None:
        savepath = f'{savepath}_class={restrict_to_class}'
    
    if attack_vectors is None:
        attack_vectors = []
        for attack in adversarial_attacks:
            attack_vectors.append([attack.compute_attack(test_points, budget) - test_points for budget in tqdm(budget_range)])
    
    for attack, attack_vec in zip(adversarial_attacks, attack_vectors):
        for budget_idx in range(0, len(budget_range), 10):
            infty_norms = torch.linalg.vector_norm(attack_vec[budget_idx].reshape(attack_vec[budget_idx].shape[0], -1), ord=float('inf'), dim=1)
            if restrict_to_class is not None:
                not_class = torch.where(attack.proba(test_points).argmax(dim=1) != restrict_to_class)
                infty_norms[not_class] = 0.
            imax = infty_norms.argmax(dim=0)

            figure, axes = plt.subplots(1, 3)
            figure.set_figwidth(15)
            attack_vec_2plot = attack_vec[budget_idx][imax]
            test_point_2plot = test_points[imax]
            proba_test_point = attack.proba(test_point_2plot).squeeze(0)
            class_test_point = torch.argmax(proba_test_point)
            proba_attack_vec = attack.proba(attack_vec_2plot).squeeze(0)
            class_attack_vec = torch.argmax(proba_attack_vec)
            proba_attacked_point = attack.proba(test_point_2plot + attack_vec_2plot).squeeze(0)
            class_attacked_point = torch.argmax(proba_attacked_point)
            
            im0 = axes[0].matshow(test_point_2plot.squeeze(0).detach(), vmax=3.5, vmin=-1)
            axes[0].set_title(f"Test point\nPredicted class: {class_test_point}\n Confidence: {proba_test_point[class_test_point]*100:.1f}%")
            colorbar(im0)
            im1 = axes[1].matshow(attack_vec_2plot.squeeze(0).detach())
            axes[1].set_title(f"Attack vector\nPredicted class: {class_attack_vec}\n Confidence: {proba_attack_vec[class_attack_vec]*100:.1f}%")
            colorbar(im1)
            im2 = axes[2].matshow((attack_vec_2plot + test_point_2plot).squeeze(0).detach(), vmax=3.5, vmin=-1)
            axes[2].set_title(f"Attacked point\nPredicted class: {class_attacked_point}\n Confidence: {proba_attacked_point[class_attacked_point]*100:.1f}%")
            colorbar(im2)
            [ax.set_axis_off() for ax in axes.ravel()]
            plt.savefig(savepath + f"_{type(attack).__name__}_worst-case_budget={budget_range[budget_idx]}.pdf")
            # plt.show()
            figure.clf()

    if plot_inf_chart:
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

def plot_attack(
    adversarial_attack: AdversarialAttack,
    number: int=1,
    savepath: str='./output/atta' 
) -> None:
    raise NotImplementedError()

def test_points_2D(nb_points, size=1):
    """Generate test points uniformly in the square [0.5-size/2, 0.5+size/2]^2."""
    return (torch.rand(nb_points, 2) - 0.5)* size + 0.5

def plot_attacks_2D(adversarial_attack, test_points, budget=0.3, color='blue'):
    """Plots the attack vectors on the input space."""        
    adversarial_attack.task = "xor"
    attack_vectors = adversarial_attack.compute_attack(test_points, budget) - test_points
    attack_vectors = attack_vectors.detach()
    for coords, attack_vector in tqdm(zip(test_points, attack_vectors)):
        plt.quiver(coords[0], coords[1], attack_vector[0], attack_vector[1], width=0.003, scale_units='xy', angles='xy', scale=1, zorder=2, color=color, label=f'{type(adversarial_attack).__name__}')
    if adversarial_attack.task == "xor":
        plt.plot([0, 1], [0, 1], "ro", zorder=3)
        plt.plot([0, 1], [1, 0], "go", zorder=3)
    elif adversarial_attack.task == "or":
        plt.plot([0], [0], "ro", zorder=3)
        plt.plot([0, 1, 1], [1, 0, 1], "go", zorder=3)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    # savepath = "./output/attacks_svd_{}_{}".format(type(adversarial_attack).__name__, adversarial_attack.task)
    # plt.savefig(savepath + '.pdf', format='pdf')
    # plt.clf()


def plot_contour_2D(adversarial_attack):
    """Plot the contour of the neural network for a 2D input manifold and 1D leaves.

    Args:
        adversarial_attack (_type_): neural network.
    """
    adversarial_attack.network = adversarial_attack.network.to(torch.double)
    xs = torch.linspace(0, 1, steps=100).double()
    grid = torch.cartesian_prod(xs, xs)
    # p = adversarial_attack.proba(grid)[..., 1]
    # p = p.reshape((*xs.shape, *xs.shape))
    # plt.pcolormesh(xs, xs, p.detach().numpy()) 
    # plt.colorbar()
    # plt.plot()
    p, p_indices = adversarial_attack.proba(grid).min(dim=-1)
    p = p.reshape((*xs.shape, *xs.shape))
    p_indices = p_indices.reshape((*xs.shape, *xs.shape))
    p[p_indices == 0] = - p[p_indices == 0]
    levels = torch.cat((-torch.logspace(int((-p[p_indices == 0]).min().log()), 0, 40), torch.logspace(int(p[p_indices == 1].min().log()), 0, 40))).sort().values
    cmap = cm.get_cmap('Spectral')
    # plt.contour(xs, xs, p.detach().numpy(), levels=levels, cmap=cmap, norm=SymLogNorm(float(p.abs().min()), float(p.abs().max()))) #, norm='symlog', vmin=-1, vmax=1)
    plt.contour(xs, xs, 1 / p.detach().numpy(), levels=(1 / levels).sort().values, cmap=cmap, norm='symlog') #, norm='symlog', vmin=-1, vmax=1)
    # plt.show()


def plot_curvature_2D(adversarial_attack):
    """Plot the extrinsic curvature for a 2D input manifold and 1D leaves.
    """
    adversarial_attack.network = adversarial_attack.network.to(torch.double)
    xs = torch.linspace(0, 1, steps=100).double()
    grid = torch.cartesian_prod(xs, xs)

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

    dx = 1e-6 * normal
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
    # plt.show()
    # plt.clf()

    """2D color plot"""
    dtheta = dtheta.reshape((*xs.shape, *xs.shape)).detach()
    plt.pcolormesh(xs, xs, dtheta, cmap='PRGn', norm=SymLogNorm(1e-6))
    # plt.pcolormesh(xs, xs, dtheta, cmap='PRGn', vmin=-dtheta.abs().max(), vmax=dtheta.abs().max())
    plt.colorbar()

    # plt.show()
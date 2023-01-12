"""Module implementing our 2 step attack."""
import random
from typing import Tuple, Union
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch import nn
from torch.autograd.functional import jacobian

class GeometricModel(object):
    
    def __init__(self,
                 network: nn.Module,
                 network_score: nn.Module,
                 verbose: bool=False,
    ) -> None:

        super(GeometricModel, self).__init__()
        self.network = network
        self.network_score = network_score
        # self.network.eval()
        self.verbose = verbose
        self.device = next(self.network.parameters()).device


    def proba(
        self,
        eval_point: torch.Tensor,
    ) -> None:

        if len(eval_point.shape) == 3:  # TODO: trouver un truc plus propre
            eval_point = eval_point.unsqueeze(0)
        p = torch.exp(self.network(eval_point))
        if self.verbose: print(f"proba: {p}")
        return p

    def score(
        self,
        eval_point: torch.Tensor,
    ) -> None:
        
        if len(eval_point.shape) == 3:  # TODO: trouver un truc plus propre
            eval_point = eval_point.unsqueeze(0)
        
        return self.network_score(eval_point)


    def grad_proba(
        self,
        eval_point: torch.Tensor,
        wanted_class: int, 
    ) -> torch.Tensor:

        j = jacobian(self.proba, eval_point).squeeze(0)

        grad_proba = j[wanted_class, :]

        return grad_proba


    def jac_proba(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:

        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.proba, eval_point) # TODO: vérifier dans le cadre non batched
        if self.verbose: print(f"shape of j before reshape = {j.shape}")
        j = j.sum(2)
        j = j.reshape(*(j.shape[:2]), -1)
        if self.verbose: print(f"shape of j after reshape = {j.shape}")

        return j
    
    
    def jac_score(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        
        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.score, eval_point)
        if self.verbose: print(f"shape of j before reshape = {j.shape}")
        
        j = j.sum(2)
        j = j.reshape(*(j.shape[:2]), -1)
        if self.verbose: print(f"shape of j after reshape = {j.shape}")
        
        return j
    
    
    def local_data_matrix(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        
        J_s = self.jac_score(eval_point)
        p = self.proba(eval_point)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("zi,zj -> zij", p, p)
        
        return torch.einsum("zji, zjk, zkl -> zil", J_s, (P - pp), J_s)


class AdversarialAttack(GeometricModel):
    """Class to represent a general adversarial attack
    and to analyse its performances."""

    def compute_attack(self, init_point, budget, *args, **kwargs):
        """Computes the attack on point init_point with
        an euclidean budget.

        :init_point: torch tensor (2)
        :budget: positive real number
        :returns: attacked point as a torch tensor (2)

        """
        raise NotImplementedError()

        
    def test_attack(self, budget, test_points):
        """Computes multiple attacks to check the fooling
        rate of the attack.

        :budget: max euclidean size of an attack

        :returns: fooling rate
        """
        
        attacked_points = self.compute_attack(test_points, budget)
        predicted_labels = self.network(test_points).exp().argmax(dim=1)
        predicted_labels_attacked = self.network(attacked_points).exp().argmax(dim=1)
        fooling_rate = (predicted_labels != predicted_labels_attacked).float().mean()

        return fooling_rate

    def test_points_2D(self, nb_points, size=1):
        """Generate test points uniformly in the square [0.5-size/2, 0.5+size/2]^2."""
        return (torch.rand(nb_points, 2) - 0.5)* size + 0.5

    def plot_attacks_2D(self, nb_test_points=int(1e2), budget=0.3):
        """Plots the attack vectors on the input space."""        
        test_points = self.test_points_2D(nb_test_points)  # maybe change this
        for coords in tqdm(test_points):
            attack_vector = self.compute_attack(coords, budget, plot=True) - coords
            plt.quiver(coords[0], coords[1], attack_vector[0], attack_vector[1], width=0.003, scale_units='xy', angles='xy', scale=1, zorder=2)
        if self.task == "xor":
            plt.plot([0, 1], [0, 1], "ro", zorder=3)
            plt.plot([0, 1], [1, 0], "go", zorder=3)
        elif self.task == "or":
            plt.plot([0], [0], "ro", zorder=3)
            plt.plot([0, 1, 1], [1, 0, 1], "go", zorder=3)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        savepath = "./output/attacks_{}_{}".format(type(self).__name__, self.task)
        plt.savefig(savepath + '.pdf', format='pdf')
        plt.show()

    def save_xor_fooling_rates(self, nb_test_points=int(5e3), step=1e-2, size=1, end=1):
        """Saves the graph of fooling rates with respect to the budget.
        :returns: None
        """
        
        test_points = self.test_points_2D(nb_test_points, size)  # maybe change this
        budget_range = torch.arange(0, end, step)
        fooling_rates = [self.test_attack(budget, test_points) for budget in tqdm(budget_range)]
        plt.plot(budget_range, fooling_rates, label=type(self).__name__)
        plt.xlabel("Budget")
        plt.ylabel("Fooling rate")
        savepath = "./output/fooling_rates_{}".format(type(self).__name__)
        plt.savefig(savepath + '.pdf', format='pdf')


    def save_fooling_rates(self, test_points, step=1e-2, end=1):
        """Saves the graph of fooling rates with respect to the budget.
        :test_point: points to compute the fooling rates on.
        :step: step size between two budgets.
        :end: max budget.
        :returns: None
        """
        
        budget_range = torch.arange(0, end, step)
        fooling_rates = [self.test_attack(budget, test_points) for budget in tqdm(budget_range)]
        plt.plot(budget_range, fooling_rates, label=type(self).__name__)
        plt.xlabel("Budget")
        plt.ylabel("Fooling rate")
        # savepath = "./output/fooling_rates_{}".format(type(self).__name__)
        # plt.savefig(savepath + '.pdf', format='pdf')
        # plt.show()

class StandardTwoStepSpectralAttack(AdversarialAttack):
    """Class to compute the two-step spectral attack in
    standard coordinates, and analyse it."""

    def compute_attack(self, input_sample, budget, plot=False):
        """Compute the attack on a point [input_sample]
        with a euclidean size given by [budget].

        :input_sample: torch tensor (bs, d)
        :budget: positive real number
        :returns: attacked point as a torch tensor (bs, d)

        """
        first_step_size = budget * 1  # TODO: fix this: should be in args or in init #

        assert 0 <= first_step_size <= budget

        """Computing first step's direction."""
        G = self.local_data_matrix(input_sample)
        e_1, v_1 = torch.linalg.eigh(G)  # value, vector, in ascending order
        first_step = v_1[..., -1]  # be careful, it isn't intuitive -> RTD
        norm = torch.linalg.vector_norm(first_step, ord=2, dim=-1, keepdim=True)
        first_step = first_step_size * first_step / norm
        first_step = first_step.reshape(input_sample.shape)

        """Computing first step's sign."""
        first_step_sign = torch.sign(-self.proba(input_sample + first_step).log().sum(1) + self.proba(input_sample).log().sum(1))  # TODO: sum, or else ?
        first_step = torch.einsum('z, z... -> z...', first_step_sign, first_step)
        # TODO: since less budget, we might go in the wrong direction ( close to the frontiers )

        if plot:
            plt.quiver(input_sample[0], input_sample[1], (first_step)[0], (first_step)[1], width=0.001, scale_units='xy', angles='xy', scale=1, zorder=3, color="blue")

        """Computing second step's direction."""
        G = self.local_data_matrix(input_sample + first_step)
        e_2, v_2 = torch.linalg.eigh(G)  # value, vector, in ascending order
        second_step = v_2[..., -1]
        norm = torch.linalg.vector_norm(second_step, ord=2, dim=-1, keepdim=True)
        second_step = (budget - first_step_size) * second_step / norm
        second_step = second_step.reshape(input_sample.shape)

        # print(first_step.T @ second_step)
        """Computing second step's sign."""
        second_step_sign = torch.einsum('zukl, zukl -> zu', first_step, second_step).sign()
        second_step = torch.einsum('zu, zukl -> zukl', second_step_sign, second_step)

        # optimal_ratio = ((second_step - first_step).T @ G @ second_step) / ((second_step + first_step).T @ G @ (second_step + first_step))
        # TODO: needs true G for optimal ratio

        if plot:
            plt.quiver(input_sample[0] + (first_step)[0], input_sample[1] + (first_step)[1], (second_step)[0], (second_step)[1], width=0.001, scale_units='xy', angles='xy', scale=1, zorder=3, color="purple")



        return input_sample + (first_step + second_step)

class OneStepSpectralAttack(AdversarialAttack):
    """One step spectral attack designed by Zhao et al."""
    
    def compute_attack(self, input_sample, budget, *args, **kwargs):
        """Compute the attack on a point [input_sample]
        with a euclidean size given by [budget].

        :input_sample: (2)
        :budget: positive real number
        :returns: attacked point as a torch tensor (2)

        """

        G = self.local_data_matrix(input_sample)
        e, v = torch.linalg.eigh(G)  # value, vector
        perturbation = v[:, :, -1]  # be careful, it isn't intuitive -> RTD
        norm = torch.linalg.vector_norm(perturbation, ord=2, dim=-1, keepdim=True)
        perturbation = budget * perturbation / norm
        perturbation = perturbation.reshape(input_sample.shape)
        """Computing first step's sign."""
        perturbation_sign = torch.sign(-self.proba(input_sample + perturbation).log().sum(1) + self.proba(input_sample).log().sum(1))  # TODO: sum, or else ?
        perturbation = torch.einsum('z, z... -> z...', perturbation_sign, perturbation)
        return input_sample + perturbation
"""Module implementing our 2 step attack."""
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from geometry import GeometricModel
from matplotlib import cm
from matplotlib.colors import SymLogNorm


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

    
    def attack_sign(
        self,
        init_point: torch.Tensor,
        perturbation: torch.Tensor,
        zero_value: float=1.,
    ) -> torch.Tensor:
        """Compute the sign of the attack to decrease the likelihood of the target.

        Args:
            init_point (torch.Tensor): point on where the attack take place with shape (bs, d)

        Returns:
            torch.Tensor: Batch tensor of the sign
        """

        max_likelihood_attacked, max_indices = torch.max(self.proba(init_point + perturbation), dim=1)
        max_likelihood_origin = self.proba(init_point).gather(1, max_indices.unsqueeze(1)).squeeze(1)

        perturbation_sign = torch.sign(max_likelihood_origin - max_likelihood_attacked)  # TODO: sum, or else ?
        perturbation_sign[perturbation_sign==0] = zero_value
        
        return perturbation_sign

        
    def test_attack(self, budget, test_points):
        """Computes multiple attacks to check the fooling
        rate of the attack.

        :budget: max euclidean size of an attack

        :returns: fooling rate
        """
        
        # print("get attacked point")
        attacked_points = self.compute_attack(test_points, budget)
        # print("get predicted label")
        predicted_labels = self.network(test_points).exp().argmax(dim=1)
        predicted_labels_attacked = self.network(attacked_points).exp().argmax(dim=1)
        # print('compute fooling rates')
        fooling_rate = (predicted_labels != predicted_labels_attacked).float().mean()

        return fooling_rate

    def test_points_2D(self, nb_points, size=1):
        """Generate test points uniformly in the square [0.5-size/2, 0.5+size/2]^2."""
        return (torch.rand(nb_points, 2) - 0.5)* size + 0.5

    def plot_attacks_2D(self, test_points, budget=0.3):
        """Plots the attack vectors on the input space."""        
        self.task = "xor"
        attack_vectors = self.compute_attack(test_points, budget) - test_points
        attack_vectors = attack_vectors.detach()
        for coords, attack_vector in tqdm(zip(test_points, attack_vectors)):
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
    
    
    def plot_curvature_2D(self):
        """Plot the extrinsic curvature for a 2D input manifold and 1D leaves.
        """

        xs = torch.linspace(0, 1, steps=100)
        grid = torch.cartesian_prod(xs, xs)
        p = self.proba(grid)[..., 1]
        p = p.reshape((*xs.shape, *xs.shape))
        plt.pcolormesh(xs, xs, p.detach().numpy()) 
        # levels = torch.logspace(-16, 0, 20)
        # plt.contour(xs, xs, p.detach().numpy(), levels=levels)
        plt.colorbar()
        plt.show()

        G_1 = self.local_data_matrix(grid)
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
        normal_sign = self.attack_sign(grid, normal)
        normal = torch.einsum('z, z... -> z...', normal_sign, normal)

        dx = 1e-3 * normal
        print(f"grid: {grid.shape} and dx: {dx.shape}")
        print(f"grid + dx: {(grid+dx).shape}")
        G_dx = self.local_data_matrix(grid + dx)

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

        dtheta = torch.asin(cross.abs())
        
        """3D plot"""
        # X, Y = torch.meshgrid(xs, xs)
        # dtheta = dtheta.reshape(X.shape)

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.plot_surface(X, Y, dtheta.detach(), cmap=cm.Blues)

        """2D color plot"""
        dtheta = dtheta.reshape((*xs.shape, *xs.shape)).detach()
        plt.pcolormesh(xs, xs, dtheta, norm=SymLogNorm(dtheta.abs().min().numpy() + 1e-12))
        plt.colorbar()

        # plt.show()

        

    # def save_xor_fooling_rates(self, nb_test_points=int(5e3), step=1e-2, size=1, end=1):
    #     """Saves the graph of fooling rates with respect to the budget.
    #     :returns: None
    #     """
        
    #     test_points = self.test_points_2D(nb_test_points, size)  # maybe change this
    #     budget_range = torch.arange(0, end, step)
    #     fooling_rates = [self.test_attack(budget, test_points) for budget in tqdm(budget_range)]
    #     plt.plot(budget_range, fooling_rates, label=type(self).__name__)
    #     plt.xlabel("Budget")
    #     plt.ylabel("Fooling rate")
    #     savepath = "./output/fooling_rates_{}".format(type(self).__name__)
    #     plt.savefig(savepath + '.pdf', format='pdf')


    def save_fooling_rates(self, test_points, step=1e-2, end=1, savepath="./output/fooling_rates"):
        """Saves the graph of fooling rates with respect to the budget.
        :test_point: points to compute the fooling rates on.
        :step: step size between two budgets.
        :end: max budget.
        :returns: None
        """
        
        budget_range = torch.arange(0, end, step).cpu()
        fooling_rates = [self.test_attack(budget, test_points).cpu() for budget in tqdm(budget_range)]
        plt.plot(budget_range, fooling_rates, label=type(self).__name__)
        plt.xlabel("Budget")
        plt.ylabel("Fooling rate")
        savepath = savepath + f"_{type(self).__name__}"
        torch.save((budget_range, fooling_rates), savepath + '_budget-rates.pt')
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
        first_step_size = budget * 0.6  # TODO: fix this: should be in args or in init #

        assert 0 <= first_step_size <= budget

        """Computing first step's direction."""
        # print("computing first direction")
        G_1 = self.local_data_matrix(input_sample)
        # print("computing eigh")
        if G_1.is_cuda:
            _, _, v_1 = torch.linalg.svd(G_1)
            first_step = v_1[..., 0]  # be careful, it isn't intuitive -> RTD
        else:
            _, v_1 = torch.linalg.eigh(G_1)  # value, vector, in ascending order
            first_step = v_1[..., -1]  # be careful, it isn't intuitive -> RTD
        norm_1 = torch.linalg.vector_norm(first_step, ord=2, dim=-1, keepdim=True)
        first_step = first_step_size * first_step / norm_1
        first_step = first_step.reshape(input_sample.shape)

        """Computing first step's sign."""
        # print("compute attack sign")
        first_step_sign = self.attack_sign(input_sample, first_step)
        first_step = torch.einsum('z, z... -> z...', first_step_sign, first_step)
        # TODO: since less budget, we might go in the wrong direction ( close to the frontiers )

        if plot:
            plt.quiver(input_sample[0], input_sample[1], (first_step)[0], (first_step)[1], width=0.001, scale_units='xy', angles='xy', scale=1, zorder=3, color="blue")

        """Computing second step's direction."""
        G_2 = self.local_data_matrix(input_sample + first_step)
        if G_2.is_cuda:
            _, _, v_2 = torch.linalg.svd(G_2)
            second_step = v_2[..., 0]  # be careful, it isn't intuitive -> RTD
        else:
            _, v_2 = torch.linalg.eigh(G_2)  # value, vector, in ascending order
            second_step = v_2[..., -1]  # be careful, it isn't intuitive -> RTD
        norm_2 = torch.linalg.vector_norm(second_step, ord=2, dim=-1, keepdim=True)
        second_step = (budget - first_step_size) * second_step / norm_2
        second_step = second_step.reshape(input_sample.shape)

        # print(first_step.T @ second_step)
        """Computing second step's sign."""
        second_step_sign = torch.einsum('z..., z... -> z', first_step, second_step).sign()
        second_step_sign[second_step_sign==0] = 1
        second_step = torch.einsum('z, z... -> z...', second_step_sign, second_step)

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
        if G.is_cuda:
            _, _, v = torch.linalg.svd(G)
            perturbation = v[..., 0]  # be careful, it isn't intuitive -> RTD
        else:
            _, v = torch.linalg.eigh(G)  # value, vector, in ascending order
            perturbation = v[..., -1]  # be careful, it isn't intuitive -> RTD
        norm = torch.linalg.vector_norm(perturbation, ord=2, dim=-1, keepdim=True)
        perturbation = budget * perturbation / norm
        perturbation = perturbation.reshape(input_sample.shape)
        perturbation_sign = self.attack_sign(input_sample, perturbation)
        perturbation = torch.einsum('z, z... -> z...', perturbation_sign, perturbation)
        return input_sample + perturbation
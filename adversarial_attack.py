"""Module implementing our 2 step attack."""
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from geometry import GeometricModel


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


    def save_attack(
        self,
        test_points: torch.tensor,
        budget_step: float=1e-2,
        budget_max: float=1.,
        savepath: str="./output/attacked_points"
        ) -> None:

        budget_range = torch.arange(0, budget_max, budget_step).cpu()
        
        attack_vectors = [self.compute_attack(test_points, budget) - test_points for budget in tqdm(budget_range)]
        torch.save((budget_range, test_points, attack_vectors), savepath + 'budget-points-attack.pt')
    
    
    def batch_save_attack(
        self,
        test_points: torch.tensor,
        batch_size: int=125,
        budget_step: float=1e-2,
        budget_max: float=1.,
        savepath: str="./output/attacked_points"
        ) -> None:
        
        for batch_index, test_points_batch in enumerate(torch.split(test_points, batch_size, dim=0)):
            self.save_attack(test_points_batch, budget_step, budget_max, f"{savepath}_{batch_index}")



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
            first_step = v_1[..., 0, :]  # be careful, it isn't intuitive -> RTD
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
            second_step = v_2[..., 0, :]  # be careful, it isn't intuitive -> RTD
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
        # _, _, v_svd = torch.linalg.svd(G)
        # perturbation_svd = v_svd[..., 0, :]  # be careful, it isn't intuitive -> RTD
        # norm_svd = torch.linalg.vector_norm(perturbation_svd, ord=2, dim=-1, keepdim=True)
        # perturbation_svd = perturbation_svd / norm_svd
        # _, v_eigh = torch.linalg.eigh(G)  # value, vector, in ascending order
        # perturbation_eigh = v_eigh[..., -1]  # be careful, it isn't intuitive -> RTD
        # norm_eigh = torch.linalg.vector_norm(perturbation_eigh, ord=2, dim=-1, keepdim=True)
        # perturbation_eigh = perturbation_eigh / norm_eigh
        # if torch.allclose(perturbation_svd.abs(), perturbation_eigh.abs()):
        #     print("RAS")
        # else:
        #     max_deviation = (perturbation_svd.abs() - perturbation_eigh.abs()).abs().max() / max(perturbation_svd.abs().max(), perturbation_eigh.abs().max())
        #     print(f"c la merde: {max_deviation}")

        if G.is_cuda:
            _, _, v = torch.linalg.svd(G)
            perturbation = v[..., 0, :]  # be careful, it isn't intuitive -> RTD
        else:
            _, v = torch.linalg.eigh(G)  # value, vector, in ascending order
            perturbation = v[..., -1]  # be careful, it isn't intuitive -> RTD
        norm = torch.linalg.vector_norm(perturbation, ord=2, dim=-1, keepdim=True)
        perturbation = budget * perturbation / norm
        perturbation = perturbation.reshape(input_sample.shape)
        perturbation_sign = self.attack_sign(input_sample, perturbation)
        perturbation = torch.einsum('z, z... -> z...', perturbation_sign, perturbation)
        return input_sample + perturbation
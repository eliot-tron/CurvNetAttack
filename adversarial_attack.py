"""Module implementing our 2 step attack."""
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from geometry import GeometricModel
from autoattack import AutoAttack

class AdversarialAttack(GeometricModel):
    """Class to represent a general adversarial attack
    and to analyse its performances."""

    def compute_attack(self, input_sample, budget, *args, **kwargs):
        """Computes the attack on point init_point with
        an euclidean budget.

        :input_sample+: torch tensor (bs, d)
        :budget: positive real number
        :returns: attacked point as a torch tensor (bs, d)

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

        max_likelihood_origin, max_indices = torch.max(self.proba(init_point), dim=1)
        # max_likelihood_attacked, max_indices = torch.max(self.proba(init_point + perturbation), dim=1)
        max_likelihood_attacked = self.proba(init_point + perturbation).gather(1, max_indices.unsqueeze(1)).squeeze(1)
        max_likelihood_attacked_neg = self.proba(init_point - perturbation).gather(1, max_indices.unsqueeze(1)).squeeze(1)

        perturbation_sign = torch.sign(max_likelihood_attacked_neg - max_likelihood_attacked)  # TODO: sum, or else ?
        perturbation_sign[perturbation_sign == 0] = zero_value
        
        return perturbation_sign

        
    def test_attack(self, budget, test_points, attack_vectors=None):
        """Computes multiple attacks to check the fooling
        rate of the attack.

        :budget: max euclidean size of an attack
        :test_points: points to attack
        :attack: if not None, will not recompute the attack and will use this instead.

        :returns: fooling rate
        """
        
        if attack_vectors is None:
            # print("get attacked point")
            attacked_points = self.compute_attack(test_points, budget)
        else:
            attacked_points = test_points + attack_vectors
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

        budget_range = torch.arange(0, budget_max, budget_step)
        
        attack_vectors = [self.compute_attack(test_points, budget).detach() - test_points for budget in tqdm(budget_range)]
        torch.save((budget_range, test_points, attack_vectors), savepath + f'_{type(self).__name__}_budget-points-attack.pt')
        del attack_vectors

class TwoStepSpectralAttack(AdversarialAttack):
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

    def save_attack(
            self,
            test_points: torch.tensor,
            budget_step: float = 0.01,
            budget_max: float = 1,
            savepath: str = "./output/attacked_points"
            ) -> None:
        """Allow to save some time by computing only one unit norm attack
        and multiplying by the budgets. This works because the direction
        of the one step attack doesn't change wrt the budget. However,
        we still need to recompute the sign of the perturbation.
            
        """

        budget_range = torch.arange(0, budget_max, budget_step).cpu()
        
        unit_norm_perturbation =  (self.compute_attack(test_points, 1.).detach() - test_points)
        attack_vectors = []
        for budget in tqdm(budget_range):
            perturbation = budget * unit_norm_perturbation
            perturbation_sign = self.attack_sign(test_points, perturbation)
            perturbation = torch.einsum('z, z... -> z...', perturbation_sign, perturbation)
            attack_vectors.append(perturbation)
        torch.save((budget_range, test_points, attack_vectors), savepath + f'_{type(self).__name__}_budget-points-attack.pt')
        del attack_vectors

        

class AdversarialAutoAttack(AdversarialAttack):
    """Auto Attack designed by ... with the AutoAttack Package

    """

    def compute_attack(self, input_sample, budget, *args, **kwargs):
        """Compute the attack on a point [input_sample]
        with a euclidean size given by [budget].

        :input_sample: torch tensor (bs, d)
        :budget: positive real number
        :returns: attacked point as a torch tensor (bs, d)

        """
        # adversary = AutoAttack(self.network_score, norm='L2', eps=budget, version='custom', attacks_to_run=['apgd-ce'], device=self.device, verbose=False)
        adversary = AutoAttack(self.network_score.float(), norm='L2', eps=budget, version='standard', device=self.device, verbose=False)
        labels = torch.argmax(self.proba(input_sample), dim=-1)
        x_adv = adversary.run_standard_evaluation(input_sample.clone().float(), labels, bs=250) 
        self.network_score.to(self.dtype) # to avoid messing with other attacks
        return x_adv.to(self.dtype)


class APGDAttack(AdversarialAttack):
    """Auto Attack designed by ... with the AutoAttack Package

    """

    def compute_attack(self, input_sample, budget, *args, **kwargs):
        """Compute the attack on a point [input_sample]
        with a euclidean size given by [budget].

        :input_sample: torch tensor (bs, d)
        :budget: positive real number
        :returns: attacked point as a torch tensor (bs, d)

        """
        adversary = AutoAttack(self.network_score.float(), norm='L2', eps=budget, version='custom', attacks_to_run=['apgd-ce'], device=self.device, verbose=False)
        labels = torch.argmax(self.proba(input_sample), dim=-1)
        x_adv = adversary.run_standard_evaluation(input_sample.clone().float(), labels, bs=250)
        self.network_score.to(self.dtype) # to avoid messing with other attacks
        return x_adv.to(self.dtype)


class GeodesicSpectralAttack(AdversarialAttack):
    """Adversarial attack computing the true geodesic with initial velocity
    the eigenvector associated with the highest eigenvalue of the FIM.
    """

    def compute_attack(self, input_sample, budget, *args, **kwargs):
        """Compute the attack on a point [input_sample]
        with a euclidean size given by [budget].

        :input_sample: torch tensor (bs, d)
        :budget: positive real number
        :returns: attacked point as a torch tensor (bs, d)

        """
        G = self.local_data_matrix(input_sample)

        if G.is_cuda:
            _, _, v = torch.linalg.svd(G)
            init_speed = v[..., 0, :]  # be careful, it isn't intuitive -> RTD
        else:
            _, v = torch.linalg.eigh(G)  # value, vector, in ascending order
            init_speed = v[..., -1]  # be careful, it isn't intuitive -> RTD
        norm = torch.linalg.vector_norm(init_speed, ord=2, dim=-1, keepdim=True)
        init_speed = budget * init_speed / norm
        init_speed = init_speed.reshape(input_sample.shape)
        init_speed_sign = self.attack_sign(input_sample, init_speed)
        init_speed = torch.einsum('z, z... -> z...', init_speed_sign, init_speed)

        x_adv = self.geodesic(input_sample, init_speed, budget)
        x_adv = input_sample + budget * torch.nn.functional.normalize((x_adv - input_sample).flatten(1), dim=1).reshape(x_adv.shape)

        return x_adv

"""Module implementing our 2 step attack."""
import matplotlib.pyplot as plt
import network as net
import torch
from tqdm import tqdm


class AdversarialAttack(object):
    """Class to represent a general adversarial attack
    and to analyse its performances."""

    def __init__(self, network, task="xor"):
        """Initialize the attack on the network."""
        super(AdversarialAttack, self).__init__()
        self.network = network
        self.network.eval()
        self.task = task

    def true_label(self, input_point):
        """From an input_point returns the true label
        depending on the task of the network.

        :input_point: batched tensor (n, 2)
        :returns: TODO

        """

        if self.task == "xor":
            rounded_input_point = torch.round(input_point, dtype=int)
            true_label = torch.logical_xor(*rounded_input_point).type(torch.float)
        elif self.task == "or":
            rounded_input_point = torch.round(input_point, dtype=int)
            true_label = torch.logical_or(*rounded_input_point).type(torch.float)
        else:
            raise NotImplementedError()

        return true_label

    def compute_attack(self, init_point, budget):
        """Computes the attack on point init_point with
        an euclidean budget.

        :init_point: torch tensor (2)
        :budget: positive real number
        :returns: torch tensor (2)

        """
        raise NotImplementedError()

        
    def test_attack(self, budget, test_points):
        """Compute multiple attacks to check the fooling
        rate of the attack.

        :budget: max euclidean size of an attack

        :returns: fooling rate
        """
        if self.task not in ["xor", "or"]:
            raise NotImplementedError()
        
        attacked_points = torch.cat([self.compute_attack(input_sample, budget).unsqueeze(0) for input_sample in test_points])
        predicted_labels = torch.round(self.network(test_points))
        predicted_labels_attacked = torch.round(self.network(attacked_points))
        fooling_rate = torch.sum(torch.abs(predicted_labels - predicted_labels_attacked)) / len(test_points)

        return fooling_rate 

    def plot_fooling_rates(self, nb_test_points=int(1e3), step=1e-2):
        """Plot the graph of fooling rates with respect to the budget.
        :returns: TODO

        """
        
        test_points = torch.rand(nb_test_points, 2)  # maybe change this
        budget_range = torch.arange(0, 1, step)
        fooling_rates = [self.test_attack(budget*0, test_points) for budget in tqdm(budget_range)]
        plt.plot(budget_range, fooling_rates)
        savepath = "./plots/fooling_rates_{}_{}".format(type(self).__name__, self.task)
        plt.savefig(savepath + '.pdf', format='pdf')
        plt.show()


class DogLegAttack(AdversarialAttack):
    """Class to compute the dog-leg attack and analyse it."""

    def compute_attack(self, input_sample, budget, first_step_size):
        """Compute the attack on points input_sample with
        a euclidean budget.

        :input_sample: torch tensor (2)
        :budget: positive real number
        :returns: torch tensor (2)

        """

        assert 0 < first_step_size < budget

        W_1 = self.network.hid_layer.weight
        b_1 = self.network.hid_layer.bias
        W_2 = self.network.out_layer.weight
        Sigma = sigmoid_prime((W_1 @ input_sample).squeeze() + b_1)
        J = (W_2 @ Sigma @ W_1)
        e_1, v_1 = torch.eig(J.T @ J, eigenvectors=True)
        imax = torch.argmax(e_1, dim=0)[0]
        first_step = v_1[:, imax]  # be careful, it isn't intuitive -> RTD
        first_step = first_step_size * first_step / first_step.norm()
        if -torch.log(self.network(input_sample + first_step)) <= -torch.log(self.network(input_sample)):  # Doesn't apply to us because only trained on {0,1}
            first_step = -first_step
        
        J_inverted = J.inv()
        R_v = NotImplemented  # = R_iklj v^k v^l with v=first_step 
        B = torch.eye(*R_v.shape) + R_v / 3.
        P_inv = NotImplemented  # from normal coordinates to standard coordinates 
        e_2, v_2 = torch.eig(J_inverted @ J_inverted.T @ B, eigenvectors=True)
        imax = torch.argmax(e_2, dim=0)[0]
        second_step = v_2[:, imax]
        second_step = P_inv @ second_step
        second_step = (budget - first_step_size) * second_step / second_step.norm()

        # TODO: How do we chose + or - for second_step? <12-01-22, Eliot> #
        if -torch.log(self.network(input_sample + first_step + second_step)) <= -torch.log(self.network(input_sample + first_step)):  # Doesn't apply to us because only trained on {0,1}
            second_step = -second_step
        
        return first_step + second_step


class OneStepSpectralAttack(AdversarialAttack):
    """One step spectral attack designed by Zhao et al."""
    
    def compute_attack(self, input_sample, budget):
        """Compute the attack on points input_sample
        with a euclidean budget.

        :input_sample: (2)
        :budget: positive real number
        :returns: Torch tensor (2)

        """

        W_1 = self.network.hid_layer.weight
        b_1 = self.network.hid_layer.bias
        W_2 = self.network.out_layer.weight
        Sigma = sigmoid_prime((W_1 @ input_sample).squeeze() + b_1)
        J = (W_2 @ Sigma @ W_1)
        e, v = torch.eig(J.T @ J, eigenvectors=True)
        imax = torch.argmax(e, dim=0)[0]
        perturbation = v[:, imax]  # be careful, it isn't intuitive -> RTD
        perturbation = budget * perturbation / perturbation.norm()
        if -torch.log(self.network(input_sample + perturbation)) <= -torch.log(self.network(input_sample)):  # Doesn't apply to us because only trained on {0,1}
            perturbation = -perturbation
        return perturbation


def sigmoid_prime(x):
    """Compute the diagonal matrix with the values
    the derivative of the sigmoid at x_i.
    :x: (N)
    :returns: (N,N)

    """
    Y = torch.sigmoid(x)
    return torch.diag(Y * (1 - Y))

"""Module implementing our 2 step attack."""
from re import A
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import network as net
import torch
from tqdm import tqdm


class AdversarialAttack(object):
    """Class to represent a general adversarial attack
    and to analyse its performances."""

    def __init__(self, network, task="xor"):
        """Initializes the attack on the network."""
        super(AdversarialAttack, self).__init__()
        self.network = network
        self.network.eval()
        self.task = task

    def true_label(self, input_point):
        """From an input_point returns the true label
        depending on the task of the network.

        :input_point: batched tensor (n, 2)
        :returns: 0 or 1

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
        if self.task not in ["xor", "or"]:
            raise NotImplementedError()
        
        attacked_points = torch.cat([self.compute_attack(input_sample, budget).unsqueeze(0) for input_sample in test_points])
        predicted_labels = torch.round(self.network(test_points))
        predicted_labels_attacked = torch.round(self.network(attacked_points))
        fooling_rate = torch.sum(torch.abs(predicted_labels - predicted_labels_attacked)) / len(test_points)

        return fooling_rate

    def test_points(self, nb_points, size=1):
        """Generate test points uniformly in the square [0.5-size, 0.5+size]^2."""
        return (torch.rand(nb_points, 2) - 0.5)* size + 0.5

    def plot_attacks(self, nb_test_points=int(1e2), budget=0.3):
        """Plots the attack vectors on the input space."""        
        test_points = self.test_points(nb_test_points)  # maybe change this
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
        savepath = "./plots/attacks_{}_{}".format(type(self).__name__, self.task)
        plt.savefig(savepath + '.pdf', format='pdf')
        plt.show()

    def plot_fooling_rates(self, nb_test_points=int(1e3), step=1e-2, size=1):
        """Plots the graph of fooling rates with respect to the budget.
        :returns: TODO

        """
        
        test_points = self.test_points(nb_test_points, size)  # maybe change this
        budget_range = torch.arange(0, 1, step)
        fooling_rates = [self.test_attack(budget, test_points) for budget in tqdm(budget_range)]
        plt.plot(budget_range, fooling_rates, label=type(self).__name__)
        plt.xlabel("Budget")
        plt.ylabel("Fooling rate")
        savepath = "./plots/fooling_rates_{}_{}".format(type(self).__name__, self.task)
        # plt.savefig(savepath + '.pdf', format='pdf')
        # plt.show()


class StandardTwoStepSpectralAttack(AdversarialAttack):
    """Class to compute the two-step spectral attack in
    standard coordinates, and analyse it."""

    def compute_attack(self, input_sample, budget, plot=False):
        """Compute the attack on a point [input_sample]
        with a euclidean size given by [budget].

        :input_sample: torch tensor (2)
        :budget: positive real number
        :returns: attacked point as a torch tensor (2)

        """
        first_step_size = budget * 0.8  # TODO: fix this: should be in args or in init #

        assert 0 <= first_step_size <= budget

        """Computing first step's direction."""
        W_1 = self.network.hid_layer.weight
        b_1 = self.network.hid_layer.bias
        W_2 = self.network.out_layer.weight
        b_2 = self.network.out_layer.bias
        Sigma = sigmoid_prime((W_1 @ input_sample).squeeze() + b_1)
        J = (W_2 @ Sigma @ W_1)  # not really J, missing a > 0 factor
        G = J.T @ J  # not really G, missing a > 0 factor
        e_1, v_1 = torch.eig(G, eigenvectors=True)  # value, vector
        imax = torch.argmax(e_1, dim=0)[0]
        first_step = v_1[:, imax]  # be careful, it isn't intuitive -> RTD
        first_step = first_step_size * first_step / first_step.norm()

        """Computing first step's sign."""
        if -self.network.log_likelihood(input_sample + first_step) <= -self.network.log_likelihood(input_sample):
            first_step = -first_step
        # TODO: since less budget, we might go in the wrong direction ( close to the frontiers )

        if plot:
            plt.quiver(input_sample[0], input_sample[1], (first_step)[0], (first_step)[1], width=0.001, scale_units='xy', angles='xy', scale=1, zorder=3, color="blue")

        """Computing second step's direction."""
        W_1 = self.network.hid_layer.weight
        b_1 = self.network.hid_layer.bias
        W_2 = self.network.out_layer.weight
        b_2 = self.network.out_layer.bias
        Sigma = sigmoid_prime((W_1 @ (input_sample + first_step)).squeeze() + b_1)
        J = (W_2 @ Sigma @ W_1)  # not really J, missing a > 0 factor
        G = J.T @ J
        e_2, v_2 = torch.eig(G, eigenvectors=True)
        imax = torch.argmax(e_2, dim=0)[0]
        second_step = v_2[:, imax]
        second_step = (budget - first_step_size) * second_step / (second_step).norm()

        # print(first_step.T @ second_step)
        """Computing second step's sign."""
        if (first_step.T @ second_step) < 0:
            second_step = -second_step

        if plot:
            plt.quiver(input_sample[0] + (first_step)[0], input_sample[1] + (first_step)[1], (second_step)[0], (second_step)[1], width=0.001, scale_units='xy', angles='xy', scale=1, zorder=3, color="purple")



        return input_sample + (first_step + second_step)


class TwoStepSpectralAttack(AdversarialAttack):
    """Class to compute the two-step spectral attack and analyse it."""

    def compute_attack(self, input_sample, budget, plot=False):
        """Compute the attack on a point [input_sample]
        with a euclidean size given by [budget].

        :input_sample: torch tensor (2)
        :budget: positive real number
        :returns: attacked point as a torch tensor (2)

        """
        first_step_size = budget * 0.8  # TODO: fix this: should be in args or in init #

        assert 0 <= first_step_size <= budget

        """Computing first step's direction."""
        W_1 = self.network.hid_layer.weight
        b_1 = self.network.hid_layer.bias
        W_2 = self.network.out_layer.weight
        b_2 = self.network.out_layer.bias
        Sigma = sigmoid_prime((W_1 @ input_sample).squeeze() + b_1)
        J = (W_2 @ Sigma @ W_1)  # not really J, missing a > 0 factor
        G = J.T @ J  # not really G, missing a > 0 factor
        e_1, v_1 = torch.eig(G, eigenvectors=True)  # value, vector
        imax = torch.argmax(e_1, dim=0)[0]
        first_step = v_1[:, imax]  # be careful, it isn't intuitive -> RTD
        first_step = first_step_size * first_step / first_step.norm()

        """Computing first step's sign."""
        if -self.network.log_likelihood(input_sample + first_step) <= -self.network.log_likelihood(input_sample):
            first_step = -first_step
        # TODO: since less budget, we might go in the wrong direction ( close to the frontiers )

        P_inv = v_1  # @ torch.diag(1 / e_1[:,0]).sqrt() TODO: que faire ?
        P = P_inv.inverse()

        """ Switching to normal coordinates."""
        first_step = P @ first_step  # / torch.sqrt(e_1[imax, 0])

        if plot:
            plt.quiver(input_sample[0], input_sample[1], (P_inv @ first_step)[0], (P_inv @ first_step)[1], width=0.001, scale_units='xy', angles='xy', scale=1, zorder=3, color="blue")

        """Computing second step's direction."""
        G_inv = G.inverse()
        C = 3
        R_v = C * (first_step.norm().pow(2) * torch.eye(len(first_step)) - torch.einsum("i,j-> ij", first_step, first_step) )  # = R_iklj v^k v^l in normal coordinates
 
        # TODO: not really torch.eye bc G is degenerate
        B = torch.eye(*R_v.shape) + R_v / 3.
        B_inv = B.inverse()
        e_2, v_2 = torch.eig(B, eigenvectors=True)
        imax = torch.argmax(e_2, dim=0)[0] # TODO what if equality ?
        second_step = v_2[:, imax]
        # print(e_2[:,0], imax, v_2)
        # print(R_v)
        second_step = (budget - first_step_size) * second_step / (P_inv @ second_step).norm()

        # print(first_step.T @ second_step)
        """Computing second step's sign."""
        if (first_step.T @ second_step) < 0:
            second_step = -second_step

        if plot:
            plt.quiver(input_sample[0] + (P_inv @ first_step)[0], input_sample[1] + (P_inv @ first_step)[1], (P_inv @ second_step)[0], (P_inv @ second_step)[1], width=0.001, scale_units='xy', angles='xy', scale=1, zorder=3, color="purple")



        return input_sample + P_inv @ (first_step + second_step)


class TwoStepSpectralGeodesicAttack(AdversarialAttack):
    """Class to compute the two-step spectral attack and analyse it."""

    def compute_attack(self, input_sample, budget):
        """Compute the attack on a point [input_sample]
        with a euclidean size given by [budget].

        :input_sample: torch tensor (2)
        :budget: positive real number
        :returns: attacked point as a torch tensor (2)

        """
        first_step_size = budget/2  # TODO: fix this: should be in args or in init #

        assert 0 <= first_step_size <= budget

        W_1 = self.network.hid_layer.weight
        b_1 = self.network.hid_layer.bias
        W_2 = self.network.out_layer.weight
        b_2 = self.network.out_layer.bias
        Sigma = sigmoid_prime((W_1 @ input_sample).squeeze() + b_1)
        J = (W_2 @ Sigma @ W_1)  # not really J, missing a > 0 factor
        G = J.T @ J  # not really G, missing a > 0 factor
        e_1, v_1 = torch.eig(G, eigenvectors=True)  # value, vector
        imax = torch.argmax(e_1, dim=0)[0]
        first_step = v_1[:, imax]  # be careful, it isn't intuitive -> RTD
        first_step = first_step_size * first_step / first_step.norm()

        """Computing first step's sign."""
        if -self.network.log_likelihood(input_sample + first_step) <= -self.network.log_likelihood(input_sample):
            first_step = -first_step
        first_step = first_step / torch.sqrt(e_1[:, 0])  # Switching to normal coordinates
        
        G_inv = G.inverse()
        R_v = NotImplemented  # torch.zeros_like(G)  # = R_iklj v^k v^l in normal coordinates
        B = torch.eye(*R_v.shape) + R_v / 3.
        e_2, v_2 = torch.eig(G_inv @ B, eigenvectors=True)
        imax = torch.argmax(e_2, dim=0)[0]
        second_step = v_2[:, imax]
        second_step = (budget - first_step_size) * second_step / second_step.norm()

        if (first_step.T @ G @ second_step) < 0:
            second_step = -second_step

        def christoffel(x):
            """Returns Gamma_ij^k tensor (i,j,k)."""
            p = self.network(x)
            Sigma = sigmoid_prime((W_1 @ x).squeeze() + b_1)
            J = (W_2 @ Sigma @ W_1)  # not really J, missing > 0 factor

            a = (W_2 @ torch.sigmoid(W_1 @ x + b_1) + b_2) - p - 0.5
            J_cov = (G_inv @ J)
            Sigma_prime = sigmoid_second((W_1 @ x).squeeze() + b_1)
            J_der = W_1.T @ torch.diag(W_2 @ Sigma_prime) @ W_1
            B = ( torch.diag(J).unsqueeze(0) + torch.diag(J).unsqueeze(0).transpose(0, 1) - torch.einsum("k,ij -> ijk", J_cov, G) )
            C = p * (1-p) * torch.einsum("ij,k -> ijk", J_der, J_cov)

            return a * B + C

        return exp(input_sample, first_step + second_step, christoffel)


class OneStepSpectralAttack(AdversarialAttack):
    """One step spectral attack designed by Zhao et al."""
    
    def compute_attack(self, input_sample, budget, *args, **kwargs):
        """Compute the attack on a point [input_sample]
        with a euclidean size given by [budget].

        :input_sample: (2)
        :budget: positive real number
        :returns: attacked point as a torch tensor (2)

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
        """Computing first step's sign."""
        if -self.network.log_likelihood(input_sample + perturbation) <= -self.network.log_likelihood(input_sample):
            perturbation = -perturbation
        return input_sample + perturbation


def sigmoid_prime(x):
    """Compute the diagonal matrix with the values
    the derivative of the sigmoid at x_i.
    :x: (N)
    :returns: (N,N)

    """
    Y = torch.sigmoid(x)
    return torch.diag(Y * (1 - Y))

def sigmoid_second(x):
    """Compute the diagonal matrix with the values
    the second derivative of the sigmoid at x_i.
    :x: (N)
    :returns: (N,N)

    """
    Y = torch.sigmoid(x)
    return torch.diag(Y * (1 - Y) * (1 - 2*Y))

def exp(init_point, init_velocity, christoffel):
    """Compute the exponential at the point p [init_point]
    and with velocity v [init_velocity].
    Otherwise said, returns Exp_p(v).*

    :init_point: (n) tensor
    :init_velocity: (n) tensor
    :christoffel: (n, n, n) tensor Gamma_ij^k
    """
    
    def eq(y, t, christoffel):
        gamma, gamma_prime = y
        dydt = [gamma_prime, - gamma_prime @ ( gamma_prime @ christoffel(gamma) )]
        return dydt

    t = torch.linspace(0, 1, 101)
    sol = odeint(eq, [init_point, init_velocity], t, args=(christoffel))

    return sol[-1]
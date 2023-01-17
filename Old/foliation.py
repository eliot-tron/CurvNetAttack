"""Various tools to compute the foliation of
a simple 1-hidden layered neural network."""
from ast import Not
from cmath import log
import network as net
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class Foliation(object):
    """Class representing a foliation."""
    def __init__(self, network, task="xor"):
        super(Foliation, self).__init__()
        self.network = network
        self.task = task


    def compute_leaf(self, init_point, num_points=4000, dt=1e-3, transverse=False):
        """Compute the leaf going through the point
        [init_point] and with distribution the kernel
        of the FIM induced by the network.

        :init_point: (x_1(0), x_2(0))
        :num_points: number of points to generate on the curve
        :dt: time interval to solve the PDE system
        :transverse: If true, compute the transverse foliation
        :returns: the curve (x_1(t), x_2(t)) [num_points, 2]

        """
        gamma_list = [torch.tensor(init_point).unsqueeze(-1)]
        if transverse:
            P = torch.eye(2)
        else:
            P = torch.tensor([[0., -1.], [1., 0.]])
        W_1 = self.network.hid_layer.weight
        b_1 = self.network.hid_layer.bias
        W_2 = self.network.out_layer.weight

        for i in range(int(num_points/2)):
            gamma = gamma_list[i]
            Sigma = sigmoid_prime((W_1 @ gamma).squeeze() + b_1)
            gamma_list.append(gamma + dt * (W_2 @ Sigma @ W_1 @ P).T)
        
        for i in range(int(num_points/2)):
            gamma = gamma_list[0]
            Sigma = sigmoid_prime((W_1 @ gamma).squeeze() + b_1)
            gamma_list.insert(0, gamma - dt * (W_2 @ Sigma @ W_1 @ P).T)
        

        return gamma_list

    def compute_eigenvector(self, input_sample):
        """Compute One Step Spectral Attack

        :input_sample: x going through the network
        :returns: TODO

        """
        W_1 = self.network.hid_layer.weight
        b_1 = self.network.hid_layer.bias
        W_2 = self.network.out_layer.weight
        Sigma = sigmoid_prime((W_1 @ input_sample).squeeze() + b_1)
        J = (W_2 @ Sigma @ W_1)
        e, v = torch.eig(J.T @ J, eigenvectors=True)
        #  print("e = {}\nv = {}".format(e, v))
        imax = torch.argmax(e, dim=0)[0]
        #  print("imax = {}".format(imax))
        perturbation = v[:, imax]  # be careful, it isn't intuitive -> RTD
        #  print("perturbation = {}".format(perturbation))
        #  print('J^TJ = {}\nWith perturbation = {}'.format(J.T @ J, (J.T @ J).T @ perturbation))
        perturbation = perturbation / (5 * perturbation.norm())
        p = self.network(input_sample)
        p_attacked = self.network(input_sample + perturbation)
        y = torch.round(p)
        y_attacked = torch.round(p_attacked)
        # TODO: Not sure of the following
        # if self.task == 'xor':
        #     y = torch.logical_xor(*torch.round(input_sample)).type(torch.float)
        #     y_attacked = torch.logical_xor(*torch.round(input_sample + perturbation)).type(torch.float)
        # elif self.task == 'or':
        #     y = torch.logical_or(*torch.round(input_sample)).type(torch.float)
        #     y_attacked = torch.logical_or(*torch.round(input_sample + perturbation)).type(torch.float)
        # else:
        #     raise NotImplementedError()
        if -log_likelihood(y_attacked, p_attacked) <= -log_likelihood(y, p):
            perturbation = -perturbation
        return perturbation, e[imax]

    def plot(self, leaves=True, eigenvectors=True, transverse=False):
        """Plot the leaves on the input space.

        :returns: None

        """
        scale = 0.2
        if leaves:
            print("Plotting the leaves...")
            for x in tqdm(torch.arange(-0.5, 1.5 + scale, scale)):
                for y in torch.arange(-0.5, 1.5 + scale, scale):
                    coordinates = self.compute_leaf([x, y], transverse=transverse)
                    xs = [g[0] for g in coordinates]
                    ys = [g[1] for g in coordinates]
                    plt.plot(xs, ys, "b-", zorder=1)
        if eigenvectors:
            print("Plotting the eigenvectors...")
            for x in tqdm(torch.arange(0, 1 + scale, scale)):
                for y in torch.arange(0, 1 + scale, scale):
                    eigvect, eigval = self.compute_eigenvector(torch.tensor([x,y]))
                    scaling = torch.sigmoid(0.5 / eigval.norm()).item()
                    #  scaling = 50 / (torch.linalg.norm(eigval)).item()
                    #  xs = [x, x + ev[0] * scale]
                    #  ys = [y, y + ev[1] * scale]
                    plt.quiver(x, y, eigvect[0], eigvect[1], width=0.003, zorder=2)
                    # plt.quiver(x, y, -eigvect[0], -eigvect[1], scale=scaling, width=0.003, zorder=2)

        if self.task == "xor":
            plt.plot([0, 1], [0, 1], "ro", zorder=3)
            plt.plot([0, 1], [1, 0], "go", zorder=3)
        elif self.task == "or":
            plt.plot([0], [0], "ro", zorder=3)
            plt.plot([0, 1, 1], [1, 0, 1], "go", zorder=3)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        savepath = "./plots/foliation_{}neur_{}".format(self.network.hid_size, self.task)
        if eigenvectors:
            savepath = savepath + '_eigen'
        if transverse:
            savepath = savepath + '_trans'
        plt.savefig(savepath + '.pdf', format='pdf')
        plt.show()



def sigmoid_prime(x):
    """Compute the diagonal matrix with the values
    the derivative of the sigmoid at x_i.
    :x: (N)
    :returns: (N,N)

    """
    Y = torch.sigmoid(x)
    return torch.diag(Y * (1 - Y))


def log_likelihood(y, p):
    """Returns ln P(y|x)"""
    return y * torch.log(p) + (1 - y) * torch.log(1 - p)
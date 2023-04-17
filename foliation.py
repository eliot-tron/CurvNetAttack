"""Various tools to compute the foliation of
a simple 1-hidden layered neural network."""
from tokenize import Double
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from geometry import GeometricModel
from scipy.linalg import null_space
from scipy.integrate import solve_ivp


class Foliation(GeometricModel):
    """Class representing a foliation."""

    def compute_leaf(self, init_point, transverse=False):
        """Compute the leaf going through the point
        [init_points] and with distribution the kernel
        of the FIM induced by the network.

        :init_point: point from the leaf with shape (bs, d)
        :num_points: number of points to generate on the curve
        :dt: time interval to solve the PDE system
        :transverse: If true, compute the transverse foliation
        :returns: the curve gamma(t) with shape (bs, n, d)

        """
        def f(t, y):
            J = self.jac_proba(torch.tensor(y).float().unsqueeze(0)).squeeze(0).detach()
            a, b = J[0]
            if transverse:
                e = torch.tensor([a, b])
            else:
                e = torch.tensor([b, -a])
            # e = J[0]
            e = e / e.norm(2)
            # e = null_space(J) 
            # if len(e[0]) == 0:
            #     print(e, J)
            #     e = torch.tensor([0.,0.]).numpy()
            # else:
            #     e = e[:,0]
            return e.numpy()

        leaf = solve_ivp(f, t_span=(0, 0.5), y0=init_point, method='RK23').y
        leaf_back = solve_ivp(f, t_span=(0, -0.5), y0=init_point, method='RK23').y
        # print(f'leaf shape = {leaf.shape}') 
        
        return torch.cat((torch.tensor(leaf_back).flip(1)[:,:-1], torch.tensor(leaf)), dim=1)
        # return leaf

    def plot(self, leaves=True, eigenvectors=True, transverse=False):
        """Plot the leaves on the input space.

        :returns: None

        """
        self.task = "xor"
        scale = 0.1
        if leaves:
            print("Plotting the leaves...")
            xs = torch.arange(0, 1.5 + scale, scale)
            initial_coordinates = torch.cartesian_prod(xs, xs)
            for y0 in tqdm(initial_coordinates):
                try:
                    leaves = self.compute_leaf(y0, transverse=transverse)
                    plt.plot(leaves[0], leaves[1], ":", color='blue', zorder=1)
                except ValueError:
                    J = self.jac_proba(torch.tensor(y0).float().unsqueeze(0)).squeeze(0).detach()
                    print(f'Failed with {y0} and \njacobian {J}')
                    
                # print(leaves)
                # print(f'leaves shape: {leaves.shape}')

        if self.task == "xor":
            plt.plot([0, 1], [0, 1], "ro", zorder=3)
            plt.plot([0, 1], [1, 0], "go", zorder=3)
        elif self.task == "or":
            plt.plot([0], [0], "ro", zorder=3)
            plt.plot([0, 1, 1], [1, 0, 1], "go", zorder=3)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        savepath = "./output/foliation_{}".format(self.task)
        if eigenvectors:
            savepath = savepath + '_eigen'
        if transverse:
            savepath = savepath + '_trans'
        # plt.savefig(savepath + '.pdf', format='pdf')
        # plt.show()
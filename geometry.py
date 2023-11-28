"""Module implementing tools to examine the geometry of a model."""
import torch
from torch import nn
from torch.autograd.functional import jacobian, hessian
from tqdm import tqdm
# from scipy.integrate import solve_ivp
from torchdiffeq import odeint, odeint_event


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
        self.dtype = next(self.network.parameters()).dtype


    def proba(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:

        if len(eval_point.shape) == 3:  # TODO: trouver un truc plus propre
            eval_point = eval_point.unsqueeze(0)
        p = torch.exp(self.network(eval_point))
        if self.verbose: print(f"proba: {p}")
        return p

    def score(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        
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


    def jac_proba_true_xor(self, x):
        W_1 = self.network[0].weight
        b_1 = self.network[0].bias
        W_2 = self.network[2].weight
        p = self.proba(x)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("...i, ...j -> ...ij", p, p)
        T = torch.heaviside(x @ W_1.T + b_1, torch.zeros_like(b_1))
        return torch.einsum(
            "...ik, ...kh, ...h, ...hj -> ...ij",
            P - pp, W_2, T, W_1
        )

    def jac_proba(
        self,
        eval_point: torch.Tensor,
        create_graph: bool=False
    ) -> torch.Tensor:
        """Function computing the matrix ∂_l p_a 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            create_graph (bool, optional): If ``True``, the Jacobian will be
                computed in a differentiable manner.

        Returns:
            torch.Tensor: tensor ∂_l p_a with dimensions (bs, a, l)
        """

        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.proba, eval_point, create_graph=create_graph) # TODO: vérifier dans le cadre non batched
        if self.verbose: print(f"shape of j before reshape = {j.shape}")
        j = j.sum(2)  # 2 is the batch dimension for dx when the output of the net is (bs, c) and because there is no interactions between batches in the derivative we can sum over this dimension to retrieve the only non zero components.
        j = j.flatten(2)
        if self.verbose: print(f"shape of j after reshape = {j.shape}")

        return j
    
    def jac_score(
        self,
        eval_point: torch.Tensor,
        create_graph: bool=False
    ) -> torch.Tensor:
        """Function computing the matrix ∂_l s_a 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            create_graph (bool, optional): If ``True``, the Jacobian will be
                computed in a differentiable manner.

        Returns:
            torch.Tensor: tensor ∂_l s_a with dimensions (bs, a, l)
        """

        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.score, eval_point, create_graph=create_graph)
        if self.verbose: print(f"shape of j before reshape = {j.shape}")
        
        j = j.sum(2)  # 2 is the batch dimension for dx when the output of the net is (bs, c) and because there is no interactions between batches in the derivative we can sum over this dimension to retrieve the only non zero components.
        j = j.flatten(2)
        if self.verbose: print(f"shape of j after reshape = {j.shape}")
        
        return j

        
    def test_jac_proba(
        self,
        eval_point: torch.Tensor,
    ) -> None:
        J_true = self.jac_proba_true_xor(eval_point)
        J = self.jac_proba(eval_point)
        p = self.proba(eval_point)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("...i, ...j -> ...ij", p, p)
        J_from_score = torch.einsum(
            "...ik, ...kj -> ...ij",
            P - pp, self.jac_score(eval_point)
        )
        good_estimate = torch.isclose(J, J_true).all()
        print(f"Is jac_proba a good estimate for the jacobian?\
                {'Yes' if good_estimate else 'No'}\n \
                Error mean = {(J_true-J).abs().mean()}\n \
                Max error = {(J_true-J).abs().max()} out of {max(J_true.abs().max(), J.abs().max())}")
        
        good_estimate = torch.isclose(J, J_from_score).all()
        print(f"Is jac_from_score a good estimate for the jacobian?\
                {'Yes' if good_estimate else 'No'}\n \
                Error mean = {(J_from_score-J).abs().mean()}\n \
                Max error = {(J_from_score-J).abs().max()} out of {max(J_from_score.abs().max(), J.abs().max())}")

    
    def local_data_matrix(
        self,
        eval_point: torch.Tensor,
        create_graph: bool=False,
        regularisation: bool=True
    ) -> torch.Tensor:
        """Function computing the Fisher metric wrt the input of the network. 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            create_graph (bool, optional): If ``True``, the Jacobian will be
                computed in a differentiable manner.

        Returns:
            torch.Tensor: tensor g_ij with dimensions (bs, i, j).
        """
        
        J_s = self.jac_score(eval_point, create_graph=create_graph)
        p = self.proba(eval_point)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("zi,zj -> zij", p, p)
        
        G = torch.einsum("zji, zjk, zkl -> zil", J_s, (P - pp), J_s)
        
        if not regularisation:
            return G

        eps = torch.linalg.eigh(G)[0].mean(dim=-1)
        epsI = torch.einsum("z, ij -> zij", eps, torch.eye(G.shape[-1]))
        
        return G + epsI
    

    def hessian_gradproba(
        self, 
        eval_point: torch.Tensor,
        method: str= 'torch_hessian' # 'relu_optim', 'double_jac', 'torch_hessian'
    ) -> torch.Tensor:
        """Function computing H(p_a)𝛁p_b 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.
            method (str): Method to compute the hessian:
                - relu_optim: to use only if ReLU network.
                - double_jac: uses double jacobian (slow).
                - torch_hessian: uses torch.autograd.functional.hessian (less slow).

        Returns:
            torch.Tensor: Tensor (H(p_a)𝛁p_b)_l with dimensions (bs, a,b,l).
        """

        if method == 'double_jac':
            J_p = self.jac_proba(eval_point)
            def J(x): return self.jac_proba(x, create_graph=True)
            H_p = jacobian(J, eval_point).sum(3).flatten(3)  # 3 is the batch dimension for dx when the output of the net is (bs, c) and because there is no interactions between batches in the derivative we can sum over this dimension to retrieve the only non zero components.
            h_grad_p = torch.einsum("zalk, zbk -> zabl", H_p, J_p)
            return  h_grad_p

        elif method == 'torch_hessian':
            J_p = self.jac_proba(eval_point)
            shape = self.proba(eval_point).shape
            H_p = []
            for bs, point in enumerate(eval_point):
                H_list = []
                for class_index in tqdm(range(shape[1])):
                    h_p_i = hessian(lambda x: self.proba(x)[0, class_index], point)
                    h_p_i = h_p_i.flatten(len(point.shape))
                    h_p_i = h_p_i.flatten(end_dim=-2)
                    H_list.append(h_p_i)
                H_p.append(torch.stack(H_list))
            H_p = torch.stack(H_p)
            # H_list = torch.stack([torch.stack([hessian(lambda x: self.proba(x)[bs, i], eval_point[bs]) for i in range(shape[1])]) for bs in range(shape[0])])
            h_grad_p = torch.einsum("zalk, zbk -> zabl", H_p, J_p)
            return  h_grad_p
            
        elif method == 'relu_optim':
            J_p = self.jac_proba(eval_point)
            J_s = self.jac_score(eval_point)
            P = self.proba(eval_point)
            C = P.shape[-1]
            I = torch.eye(C).unsqueeze(0)
            N = P.unsqueeze(-2).expand(-1, C, -1)
            
            """Compute """
            first_term = torch.einsum("zbi, zki, zak, zal -> zabl", J_p, J_s, (I-N), J_p) 
            
            """Compute """
            second_term = torch.einsum("za, zbi, zki, zkl -> zabl", P, J_p, J_s, J_p )
            
            return first_term - second_term

    
    
    def lie_bracket(
        self,
        eval_point: torch.Tensor,
        approximation: bool=False,
    ) -> torch.Tensor:
        """Function computing [𝛁p_a, 𝛁p_b] = H(p_b)𝛁p_a - H(p_a)𝛁p_b

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor [𝛁p_a, 𝛁p_b]_l with dimensions (bs, a,b,l)
        """

        if approximation:
            J_x = self.jac_proba(eval_point)
            new_point = eval_point.unsqueeze() + J_x
            NotImplemented
        
        H_grad = self.hessian_gradproba(eval_point)
        
        return H_grad.transpose(-2, -3) - H_grad
    

    def jac_dot_product(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing ∂_i(∇p_a^t ∇p_b).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor ∂_i(∇p_a^t ∇p_b) with dimensions (bs, a, b, i).
        """

        H_grad = self.hessian_gradproba(eval_point)

        return H_grad.transpose(-2, -3) + H_grad
    

    def jac_metric(
        self,
        eval_point: torch.Tensor,
        relu_optim: bool=False,  # Else intractable
    ) -> torch.Tensor:

        if relu_optim:
            J_s = self.jac_score(eval_point)
            J_p = self.jac_proba(eval_point)
            p = self.proba(eval_point)
            pdp = torch.einsum("...a, ...bi -> ...iab", p, J_p)  # p_a ∂_i p_b
            return torch.einsum(
                        "...ak, ...iab, ...bl -> ...ikl",
                        J_s, torch.diag_embed(J_p.mT) - pdp - pdp.mT, J_s
                    )
        else:
            # raise NotImplementedError
            # self.verbose=True
            def G(x): return self.local_data_matrix(x, create_graph=True)
            if self.verbose:
                print(f"shape of eval_point = {eval_point.shape}")
                print(f"shape of output = {self.proba(eval_point).shape}")
            jac_metric = jacobian(G, eval_point)
            if self.verbose: print(f"shape of j before reshape = {jac_metric.shape}")
            jac_metric = jac_metric.sum(3).flatten(3)  #TODO: to finish indices
            if self.verbose: print(f"shape of j after reshape = {jac_metric.shape}")
            # self.verbose=False
            return jac_metric


    def christoffel(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        J_G = self.jac_metric(eval_point)
        G = self.local_data_matrix(eval_point)
        G_inv = torch.linalg.pinv(G.to(torch.double), hermitian=True).to(self.dtype) # input need to be in double
        # TODO garde fou pour quand G devient nulle, ou que G_inv diverge

        return torch.einsum(
                    "...kl, ...ijl -> ...ijk",
                    G_inv, J_G + J_G.permute(0, 2, 1, 3) - J_G.permute(0, 3, 1, 2)
                ) / 2

    def geodesic(
        self,
        eval_point: torch.Tensor,
        init_velocity: torch.Tensor,
        euclidean_budget: float=None,
        full_path: bool=False,
    ) -> torch.Tensor:
        if len(init_velocity.shape) > 2:
            init_velocity = init_velocity.flatten(1)
        
        def ode(t, y):
            x, v = y
            christoffel = self.christoffel(x)
            return (v.reshape(x.shape), -torch.einsum("...i, ...j, ...ijk -> ...k", v, v, christoffel))
        
        if euclidean_budget is None:
            y0 = (eval_point, init_velocity) # TODO: wrong dim after bash -> should be flatten ?

            solution_ode = odeint(ode, y0, t=torch.linspace(0., 4., 1000), method="rk4")
            solution_ode_x, solution_ode_v = solution_ode

            return solution_ode_x[-1]

        elif euclidean_budget <= 0.:
            return eval_point

        else:
            def euclidean_stop(t, y):
                x = y[0]
                return nn.functional.relu(euclidean_budget - torch.norm(x - eval_point, dim=1))
            if self.verbose:
                print(f"eval_point: {eval_point.shape}")
                print(f"init_velocity: {init_velocity.shape}")
            y0 = (eval_point, init_velocity)
                
            # solution_ivp = solve_ivp(ode, t_span = (0, 2), y0=(eval_point.detach().numpy(), init_velocity.detach().numpy()), method='RK23', events=euclidean_stop if euclidean_budget is not None else None)

            # event_t, solution_ode = odeint_event(ode, y0, t0=torch.tensor(0.), event_fn=euclidean_stop, atol=euclidean_budget / 100, method="rk4", options={"step_size": euclidean_budget / 1000})
            
            # if self.verbose: print(f"event_t: {event_t}")

            solution_ode = odeint(ode, y0, t=torch.linspace(0., 4., int(100 / euclidean_budget) ), method="rk4")
            
            # self.verbose = True
            solution_ode_x, solution_ode_v = solution_ode
            if full_path:
                return solution_ode_x.transpose(0, 1)
            
            if self.verbose:
                print(f"solution_ode_x: {solution_ode_x.shape}")
                print(f"solution_ode_v: {solution_ode_v.shape}")
                print(f"0 is initial value ? {torch.allclose(solution_ode_x[0], eval_point)} dist: {torch.dist(solution_ode_x[0], eval_point)}")

            # Get last point exceeding the euclidean budget
            admissible_indices = ((solution_ode_x - eval_point.unsqueeze(0)).norm(dim=-1) <= euclidean_budget)
            last_admissible_index = admissible_indices.shape[0] - 1 - admissible_indices.flip(dims=[0]).int().argmax(dim=0)
            last_admissible_solution_x = torch.diagonal(solution_ode_x[last_admissible_index]).T
                
            if self.verbose:
                last_admissible_solution_x_loop = torch.zeros_like(eval_point)
                last_admissible_index_loop = torch.zeros(eval_point.shape[0])

                for i, step in enumerate(solution_ode_x):
                    for j, batch in enumerate(step):
                        if (batch - eval_point[j]).norm() <= euclidean_budget:
                            last_admissible_index_loop[j] = i
                            last_admissible_solution_x_loop[j] = batch
                print(f"2 solutions are the same ? {torch.allclose(last_admissible_solution_x, last_admissible_solution_x_loop)}")
                print(f"2 indices of solutions are the same ? {torch.allclose(last_admissible_index.int(), last_admissible_index_loop.int())}")
                        
            return last_admissible_solution_x
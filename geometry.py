"""Module implementing tools to examine the geometry of a model."""
import torch
from torch import nn
from torch.autograd.functional import jacobian, hessian
from tqdm import tqdm


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
        create_graph: bool=False
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
        
        return torch.einsum("zji, zjk, zkl -> zil", J_s, (P - pp), J_s)
    

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
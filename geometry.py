"""Module implementing our 2 step attack."""
import torch
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
    ) -> torch.Tensor:
        
        J_s = self.jac_score(eval_point)
        p = self.proba(eval_point)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("zi,zj -> zij", p, p)
        
        return torch.einsum("zji, zjk, zkl -> zil", J_s, (P - pp), J_s)

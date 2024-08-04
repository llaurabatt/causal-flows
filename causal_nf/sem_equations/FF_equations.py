
import torch
import torch.nn.functional as F

from causal_nf.sem_equations.sem_base import SEM


class SimpleFrugalModel(SEM):
    def __init__(self, sem_name="dummy"):
        functions = None
        inverses = None

        if sem_name == "dummy":
            functions = [
                lambda *args: args[-1],  # Z1
                lambda *args: args[-1],  # Z2
                lambda *args: args[-1],  # Z3
                lambda *args: args[-1],  # Z4
                lambda *args: args[-1],  # X
                lambda *args: args[-1],  # Y
            ]
            inverses = [
                lambda *args: args[-1],  # u1
                lambda *args: args[-1],  # u2
                lambda *args: args[-1],  # u3
                lambda *args: args[-1],  # u4
                lambda *args: args[-1],  # u5
                lambda *args: args[-1],  # u6
            ]

        super().__init__(functions, inverses, None)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((6, 6))
        adj[0, :] = torch.tensor([0, 0, 0, 0, 0, 0]) # Z1
        adj[1, :] = torch.tensor([1, 0, 0, 0, 0, 0]) # Z2
        adj[2, :] = torch.tensor([1, 1, 0, 0, 0, 0]) # Z3
        adj[3, :] = torch.tensor([1, 1, 1, 0, 0, 0]) # Z4
        adj[4, :] = torch.tensor([1, 1, 1, 1, 0, 0]) # X
        adj[5, :] = torch.tensor([1, 1, 1, 1, 1, 0]) # Y

        if add_diag:
            adj += torch.eye(6)

        return adj

    def intervention_index_list(self):
        return [0, 4]
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_dense_adj

class Contrast(nn.Module):
    def __init__(self, out_dim, tau, keys):
        super(Contrast, self).__init__()
        self.proj = nn.ModuleDict({k: nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim)
        ) for k in keys})
        self.tau = tau
        for k, v in self.proj.items():
            for model in v:
                if isinstance(model, nn.Linear):
                    nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True) #按-1维度求1范数
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        # sim_matrix = torch.exp(dot_denominator / self.tau)
        return sim_matrix

    def compute_loss(self, feat1, feat2):
        z_1 = self.proj['graph'](feat1)
        z_2 = self.proj['noise_graph'](feat2)
        # z_1 = feat1
        # z_2 = feat2

        matrix_12 = self.sim(z_1,z_2)
        matrix_21 = matrix_12.t()

        pos_12 = matrix_12.diag()
        pos_21 = matrix_21.diag()

        loss_12 = -torch.log(pos_12/matrix_12.sum(1)).mean()
        loss_21 = -torch.log(pos_21/matrix_21.sum(1)).mean()
        loss = loss_12 + loss_21

        return loss

    def forward(self, feat1, feat2):
        return self.compute_loss(feat1, feat2)
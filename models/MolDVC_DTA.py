import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv
from collections import OrderedDict
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_geometric.utils import degree
from .contrastive_learning import Contrast


class Conv1dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)


class TargetRepresentation(nn.Module):
    def __init__(self, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.CNN_block = StackCNN(layer_num=3, in_channels=embedding_num, out_channels=96, kernel_size=3)

        self.linear = nn.Linear(96, embedding_num)

    def forward(self, data):
        seq_x = self.embed(data.seq_feat).permute(0, 2, 1)
        feats = self.CNN_block(seq_x)
        seq_x = self.linear(feats)
        return seq_x


class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx
class NG_GCNConv(nn.Module):
    def __init__(self, in_dim,out_dim,n_iter):
        super().__init__()
        gcn_layers = []
        for i in range(n_iter):
            gcn_layer = GCNConv(in_dim, out_dim)
            gcn_layers.append(gcn_layer)
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.att = GlobalAttentionPool(out_dim)
        self.a = nn.Parameter(torch.zeros(1, out_dim, n_iter))
        self.lin_gout = nn.Linear(out_dim, out_dim)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))
        self.relu = nn.ReLU()
        glorot(self.a)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x,edge_index,batch):
        out_list = []
        gout_list = []
        for gcn_layer in self.gcn_layers:
            gcn_out = gcn_layer(x, edge_index)
            gcn_out = self.relu(gcn_out)
            gout = self.att(gcn_out, edge_index, batch)
            out_list.append(gcn_out)
            gout_list.append(F.tanh((self.lin_gout(gout))))
        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        scores = torch.softmax(scores, dim=-1)
        scores = scores.repeat_interleave(degree(batch, dtype=batch.dtype), dim=0)
        out = (out_all * scores).sum(-1)
        return out
class LG_GCNConv(nn.Module):
    def __init__(self, in_dim,out_dim,n_iter):
        super().__init__()
        gcn_layers = []
        for i in range(n_iter):
            gcn_layer = GCNConv(in_dim, out_dim)
            gcn_layers.append(gcn_layer)
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.att = GlobalAttentionPool(out_dim)
        self.a = nn.Parameter(torch.zeros(1, out_dim, n_iter))
        self.lin_gout = nn.Linear(out_dim, out_dim)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))
        self.Prelu = nn.PReLU()
        self.relu = nn.ReLU()
        glorot(self.a)
        self.dropout = nn.Dropout(0.2)

    def forward(self, edge_attr,line_graph_edge_index,edge_index_batch):
        gcn_out = edge_attr
        out_list = []
        gout_list = []
        for gcn_layer in self.gcn_layers:
            gcn_out = gcn_layer(gcn_out, line_graph_edge_index)
            gcn_out = self.relu(edge_attr + gcn_out)
            gcn_out = self.dropout(gcn_out)
            gout = self.att(gcn_out, line_graph_edge_index, edge_index_batch)
            out_list.append(gcn_out)
            gout_list.append(F.tanh((self.lin_gout(gout))))
        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        scores = torch.softmax(scores, dim=-1)
        scores = scores.repeat_interleave(degree(edge_index_batch, dtype=edge_index_batch.dtype), dim=0)
        out = (out_all * scores).sum(-1)
        return out
class LG_Encoder(nn.Module):
    def __init__(self, edge_dim, n_feats, n_iter):
        super().__init__()
        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)
        self.LG_GCN = LG_GCNConv(n_feats, n_feats,n_iter)
        self.att = GlobalAttentionPool(n_feats)
        self.Contrast = Contrast(n_feats, 0.5, ['graph', 'noise_graph'])
        self.lin_out = nn.Linear(n_feats * 2, n_feats, bias=False)

    def forward(self, data):
        edge_index = data.edge_index
        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3

        noise_edge_attr = edge_attr.clone()
        random_noise = torch.rand_like(noise_edge_attr)
        noise_edge_attr += torch.sign(noise_edge_attr) * F.normalize(random_noise, dim=-1) * 0.2

        out = self.LG_GCN(edge_attr,data.line_graph_edge_index,data.edge_index_batch)
        noise_out = self.LG_GCN(noise_edge_attr, data.line_graph_edge_index, data.edge_index_batch)

        e_loss = self.Contrast(self.att(out, data.line_graph_edge_index, data.edge_index_batch),self.att(noise_out, data.line_graph_edge_index, data.edge_index_batch))
        out = self.lin_out(torch.cat([out,noise_out],dim=-1))
        e_g = global_add_pool(out,data.edge_index_batch)
        return e_g,e_loss
class NG_Encoder(nn.Module):
    def __init__(self, n_feats, n_iter):
        super().__init__()
        self.NG_GCN = NG_GCNConv(n_feats, n_feats,n_iter)
        self.att = GlobalAttentionPool(n_feats)
        self.Contrast = Contrast(n_feats, 0.5, ['graph', 'noise_graph'])
        self.lin_out = nn.Linear(n_feats * 2, n_feats, bias=False)

    def forward(self, data):
        x = data.x
        noise_x = x.clone()
        random_noise = torch.rand_like(noise_x)
        noise_x += torch.sign(noise_x) * F.normalize(random_noise, dim=-1) * 0.2

        out = self.NG_GCN(x,data.edge_index,data.batch)
        noise_out = self.NG_GCN(noise_x, data.edge_index,data.batch)

        n_loss = self.Contrast(self.att(out, data.edge_index, data.batch),self.att(noise_out, data.edge_index, data.batch))
        out = self.lin_out(torch.cat([out,noise_out],dim=-1))
        n_g = global_mean_pool(out,data.batch)
        return n_g,n_loss

class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=128, n_iter=3):
        super().__init__()

        self.x_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.lin_out = nn.Linear(hidden_dim*2, hidden_dim)
        self.line_graph_encoder = LG_Encoder(edge_in_dim, hidden_dim, n_iter)
        self.node_graph_encoder = NG_Encoder(hidden_dim, n_iter)

    def forward(self, data):
        data.x = self.x_mlp(data.x)
        e_g,e_loss = self.line_graph_encoder(data)
        n_g,n_loss = self.node_graph_encoder(data)
        g = self.lin_out(torch.cat([e_g,n_g],dim=-1))
        return g,n_loss,e_loss


class MolDVC_DTA(nn.Module):
    def __init__(self, vocab_protein_size = 26, embedding_size=128, out_dim=1,Alpha=0.01,Beta=0.01,n_iter=3):
        super().__init__()
        self.Alpha = Alpha
        self.Beta = Beta
        self.protein_encoder = TargetRepresentation(vocab_protein_size, embedding_size)
        self.drug_encoder = DrugEncoder(78, 6, embedding_size, n_iter=n_iter)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embedding_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_size * 2, out_dim)
        )

        self.molFC1 = nn.Linear(embedding_size, 1024)
        self.molFC2 = nn.Linear(1024, embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)


    def forward(self, data):
        protein_x = self.protein_encoder(data)
        ligand_x,n_loss,e_loss = self.drug_encoder(data)


        ligand_x = self.dropout(self.relu(self.molFC1(ligand_x)))
        ligand_x = self.dropout(self.molFC2(ligand_x))

        x = torch.cat([protein_x, ligand_x], dim=-1)
        x = self.classifier(x)
        CL_loss = n_loss * self.Alpha + e_loss * self.Beta
        return x,CL_loss




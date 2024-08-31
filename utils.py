# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年03月26日
"""
import numpy as np
from torch_geometric.data import Batch,InMemoryDataset
# from torch_geometric.data.dataset import Dataset, IndexType
from torch.utils.data import Dataset, DataLoader
from torch_geometric import data as DATA
import torch
import pickle
from torch_geometric.loader import DataLoader as pyG_DataLoader



# def collate(data):
#     drug_batch = Batch.from_data_list( [item[0] for item in data], follow_batch=['edge_index'])
#     seq_batch = Batch.from_data_list([item[1] for item in data], follow_batch=['edge_index'])
#     return drug_batch,seq_batch
def collate(data):
    data_batch = Batch.from_data_list(data,follow_batch=['edge_index'])
    return data_batch

def collate2(data):
    seq_batch = Batch.from_data_list([item for item in data])
    return seq_batch

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=collate, **kwargs)

def get_keys(d, value):
    for k, v in d.items():
        if v == value:
            return k

class DTADataset(Dataset):
    def __init__(self, smile_list, seq_list, label_list, mol_data=None, pro_data=None):
        super(DTADataset,self).__init__()
        self.smile_list = smile_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.smile_graph = mol_data
        self.pro_data = pro_data

    def __len__(self):
        return len(self.smile_list)

    def __getitem__(self, index):
        smile = self.smile_list[index]
        seq = self.seq_list[index]
        labels =self.label_list[index]

        drug_data = self.smile_graph[smile]
        _,pro_seq_feat = self.pro_data[seq]

        GCNData = DATA.Data(x=drug_data.x, edge_index=drug_data.edge_index,edge_attr=drug_data.edge_attr,line_graph_edge_index=drug_data.line_graph_edge_index,seq_feat=pro_seq_feat,y=torch.FloatTensor([labels]))
        GCNData.__setitem__('c_size', torch.LongTensor([drug_data.c_size]))
        return GCNData

class CPIDataset(Dataset):
    def __init__(self,smile_list, seq_list, label_list,mol_data = None,pro_data=None):
        super(CPIDataset,self).__init__()
        self.smile_list = smile_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.smile_graph = mol_data
        self.pro_data = pro_data

    def __len__(self):
        return len(self.smile_list)

    def __getitem__(self, index):
        smile = self.smile_list[index]
        seq = self.seq_list[index]
        labels = self.label_list[index]

        drug_data = self.smile_graph[smile]
        _, pro_seq_feat = self.pro_data[seq]

        # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
        # GCNData_smile = DATA.Data(x=drug_data.x, edge_index=drug_data.edge_index,edge_attr=drug_data.edge_attr,line_graph_edge_index=drug_data.line_graph_edge_index,y=torch.FloatTensor([labels]),
        #                           complete_edge_index=drug_data.complete_edge_index,degree=drug_data.degree,subgraph_edge_index=drug_data.subgraph_edge_index, num_subgraph_nodes=drug_data.num_subgraph_nodes, subgraph_node_index=drug_data.subgraph_node_idx, subgraph_indicator_index=drug_data.subgraph_indicator)
        GCNData = DATA.Data(x=drug_data.x, edge_index=drug_data.edge_index, edge_attr=drug_data.edge_attr,
                            line_graph_edge_index=drug_data.line_graph_edge_index, seq_feat=pro_seq_feat,
                            y=torch.FloatTensor([labels]))
        # GCNData.target = pro_seq_feat
        GCNData.__setitem__('c_size', torch.LongTensor([drug_data.c_size]))
        return GCNData


class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', graph = None,index = None ,type=None):
        super(GraphDataset, self).__init__(root)
        self.type = type
        self.index = index
        self.process(graph,index)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graph,index):
        data_list = []
        count = 0
        for key in index:
            size, features, edge_index,_ = graph[key]
            # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index),graph_num = torch.LongTensor([count]))
            GCNData.__setitem__('c_size', torch.LongTensor([size]))
            count += 1
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def proGraph(graph_data, index, device):
    proGraph_dataset = GraphDataset(graph=graph_data, index=index ,type = 'pro')
    proGraph_loader = pyG_DataLoader(proGraph_dataset, batch_size=len(graph_data), shuffle=False)
    pro_graph = None
    for batchid, batch in enumerate(proGraph_loader):
        pro_graph = batch.x.to(device),batch.edge_index.to(device),batch.graph_num.to(device),batch.batch.to(device)
    return pro_graph


def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))
def mse_print(y,f):
    mse = ((y - f)**2)
    return mse
def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


class GradAAM():
    def __init__(self, model, module):
        self.model = model
        module.register_forward_hook(self.save_hook) #To register a hook on the desired layer, its main purpose is to obtain the output of a specific layer without modifying the torch network.
        self.target_feat = None

    def save_hook(self, md, fin, fout): #Hook function: Through this hook, we can acquire the input and output results of the module.
        self.target_feat = fout

    def __call__(self, data):
        self.model.eval()
        output,_ = self.model(data)
        grad = torch.autograd.grad(output, self.target_feat)[0]
        channel_weight = torch.mean(grad, dim=0, keepdim=True)
        channel_weight = normalize(channel_weight)
        weighted_feat = self.target_feat * channel_weight
        cam = torch.sum(weighted_feat, dim=-1).detach().cpu().numpy()
        cam = normalize(cam)
        # # _,_,_,p_batch = proGraph
        # output = self.model(data).view(-1)
        # mask = torch.eq(p_batch, pro_data.seq_num)
        # indexes = torch.nonzero(mask, as_tuple=False).view(-1)
        # # new_target_feat = self.target_feat[indexes]
        # grad = torch.autograd.grad(output, self.target_feat)[0]
        # grad = grad[indexes]
        # channel_weight = torch.mean(grad, dim=0, keepdim=True)
        # channel_weight = normalize(channel_weight)
        # weighted_feat = self.target_feat[indexes] * channel_weight
        # cam = torch.sum(weighted_feat, dim=-1).detach().cpu().numpy()
        # cam = normalize(cam)
        return output.detach().cpu().numpy(), cam
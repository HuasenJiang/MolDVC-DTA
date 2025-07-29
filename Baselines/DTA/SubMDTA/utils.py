import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch

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

def collate1(data):
    drug_batch = Batch.from_data_list( [item[0] for item in data], follow_batch=['edge_index'])
    seq_batch = Batch.from_data_list([item[1] for item in data], follow_batch=['edge_index'])
    return drug_batch,seq_batch

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
        pro_seq_feat = self.pro_data[seq]

        GCNData = DATA.Data(x=drug_data.x, edge_index=drug_data.edge_index,edge_attr=drug_data.edge_attr,line_graph_edge_index=drug_data.line_graph_edge_index,target_2=pro_seq_feat[0],target_3=pro_seq_feat[1],target_4=pro_seq_feat[2],y=torch.FloatTensor([labels]))
        GCNData.__setitem__('c_size', torch.LongTensor([drug_data.c_size]))
        return GCNData

class PDBDataset(Dataset):
    def __init__(self, id_list, label_list, mol_data=None, pro_data=None):
        super(PDBDataset,self).__init__()
        self.id_list = id_list
        self.label_list = label_list
        self.ligand_graph = mol_data
        self.pro_data = pro_data

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        # smile = self.smile_list[index]
        # seq = self.seq_list[index]
        id = self.id_list[index]
        labels =self.label_list[index]

        drug_data = self.ligand_graph[id]
        pro_seq_feat = self.pro_data[id]

        GCNData = DATA.Data(x=drug_data.x, edge_index=drug_data.edge_index,edge_attr=drug_data.edge_attr,line_graph_edge_index=drug_data.line_graph_edge_index,target_2=pro_seq_feat[0],target_3=pro_seq_feat[1],target_4=pro_seq_feat[2],y=torch.FloatTensor([labels]))
        GCNData.__setitem__('c_size', torch.LongTensor([drug_data.c_size]))
        return GCNData
class GraphDTADataset(Dataset):
    def __init__(self, smile_list, seq_list, label_list, mol_data=None, pro_data=None):
        super(GraphDTADataset,self).__init__()
        self.smile_list = smile_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.smile_graph = mol_data
        # self.ppi_index = ppi_index
        self.pro_data = pro_data

    def __len__(self):
        return len(self.smile_list)

    def __getitem__(self, index):
        smile = self.smile_list[index]
        seq = self.seq_list[index]
        labels =self.label_list[index]


        drug_data = self.smile_graph[smile]
        seq_size = len(seq)
        _,pro_x,pro_edge_index,pro_seq_feat = self.pro_data[seq]

        # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
        # GCNData_smile = DATA.Data(x=drug_data.x, edge_index=drug_data.edge_index,edge_attr=drug_data.edge_attr,line_graph_edge_index=drug_data.line_graph_edge_index,y=torch.FloatTensor([labels]),
        #                           complete_edge_index=drug_data.complete_edge_index,degree=drug_data.degree,subgraph_edge_index=drug_data.subgraph_edge_index, num_subgraph_nodes=drug_data.num_subgraph_nodes, subgraph_node_index=drug_data.subgraph_node_idx, subgraph_indicator_index=drug_data.subgraph_indicator)
        GCNData_smile = DATA.Data(x=drug_data.x, edge_index=drug_data.edge_index,edge_attr=drug_data.edge_attr,line_graph_edge_index=drug_data.line_graph_edge_index,y=torch.FloatTensor([labels]))
        GCNData_smile.__setitem__('c_size', torch.LongTensor([drug_data.c_size]))
        GCNData_seq = DATA.Data(x=pro_x,edge_index=pro_edge_index,seq_feat=pro_seq_feat,y=torch.FloatTensor([labels])) # The seq_index indicates the node number of the protein in the PPI graph.
        GCNData_seq.__setitem__('c_size', torch.LongTensor([seq_size]))
        return GCNData_smile, GCNData_seq

class GraphPDBDTADataset(Dataset):
    def __init__(self, id_list, label_list, mol_data=None, pro_data=None):
        super(GraphPDBDTADataset,self).__init__()
        self.id_list = id_list
        self.label_list = label_list
        self.ligand_graph = mol_data
        self.pro_data = pro_data
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id = self.id_list[index]
        labels =self.label_list[index]
        seq_size = len(id)
        drug_data = self.ligand_graph[id]
        _,pro_x,pro_edge_index,pro_seq_feat = self.pro_data[id]

        # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
        # GCNData_smile = DATA.Data(x=drug_data.x, edge_index=drug_data.edge_index,edge_attr=drug_data.edge_attr,line_graph_edge_index=drug_data.line_graph_edge_index,y=torch.FloatTensor([labels]),
        #                           complete_edge_index=drug_data.complete_edge_index,degree=drug_data.degree,subgraph_edge_index=drug_data.subgraph_edge_index, num_subgraph_nodes=drug_data.num_subgraph_nodes, subgraph_node_index=drug_data.subgraph_node_idx, subgraph_indicator_index=drug_data.subgraph_indicator)
        GCNData_smile = DATA.Data(x=drug_data.x, edge_index=drug_data.edge_index,edge_attr=drug_data.edge_attr,line_graph_edge_index=drug_data.line_graph_edge_index,y=torch.FloatTensor([labels]))
        GCNData_smile.__setitem__('c_size', torch.LongTensor([drug_data.c_size]))
        GCNData_seq = DATA.Data(x=pro_x,edge_index=pro_edge_index,seq_feat=pro_seq_feat,y=torch.FloatTensor([labels])) # The seq_index indicates the node number of the protein in the PPI graph.
        GCNData_seq.__setitem__('c_size', torch.LongTensor([seq_size]))
        return GCNData_smile, GCNData_seq


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

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt_2=None, xt_3=None,xt_4=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt_2, xt_3, xt_4, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt_2, xt_3, xt_4,y, smile_graph):
        # assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)

        # n_word = n_word

        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target_2 = xt_2[i]
            target_3 = xt_3[i]
            target_4 = xt_4[i]
            labels = y[i]
            # mask = xt_mask[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),  # 行列转置
                                y=torch.FloatTensor([labels]))


            GCNData.target_2 = torch.LongTensor([target_2])
            GCNData.target_3 = torch.LongTensor([target_3])
            GCNData.target_4 = torch.LongTensor([target_4])
            # GCNData.target_mask = torch.LongTensor([mask])
            # GCNData.n_word = torch.tensor(n_word)

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')

        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])



def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))
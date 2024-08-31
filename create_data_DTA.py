import json
import torch
from rdkit import Chem
import numpy as np
import pickle
import pandas as pd
from collections import OrderedDict
import argparse
from torch_geometric.data import Data
from tqdm import tqdm


class CustomData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        return super().__inc__(key, value, *args, **kwargs)

def atom_features(atom):
    results = (one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

    results = np.array(results).astype(np.float32)
    return results

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def edge_features(bond):
    '''
    Get bond features
    '''
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()

def generate_drug_data(mol_graph, atom_symbols):
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats]*2, dim=0) if len(edge_feats) else edge_feats
    c_size = mol_graph.GetNumAtoms()
    features = []
    for atom in mol_graph.GetAtoms():
        feature = atom_features(atom)
        features.append(torch.from_numpy( feature / sum(feature)) )
    features = torch.stack(features)

    # This is the most essential step to convert a node graph to a line graph
    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T
    new_edge_index = edge_list.T

    data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index, edge_attr=edge_feats,c_size = c_size )

    return data

def dic_normalize(dic):
    """
    Feature normalization
    :param dic: A dict for describing residue feature
    :return: Normalizied feature
    """
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }




def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target]

def seq_to_feat(seq):
    """
    Convert a protein sequence into topo graph (i.e., feature matrix and adjacency matrix )
    :param pro_id: the UniProt id of protein
    :param seq: protein sequence
    :param db: dataset name
    :return: nodenum, feature matrix, and adjacency matrix(sparse).
    """
    target = seqs2int(seq)
    target_len = 1200
    if len(target) < target_len:
        target = np.pad(target, (0, target_len - len(target)))
    else:
        target = target[:target_len]
    target = torch.LongTensor([target])
    c_size = len(seq)
    return c_size, target



def data_split(dataset):
    """
    Make dataset spliting and Convert the original data into csv format.
    No value is returned, but the corresponding csv file is eventually generated.
    :param dataset: dataset name
    :return: None
    """
    if dataset == 'Human':
        print('convert human data into 5-fold sub-dataset !')
        df = pd.read_table('data/Human/Human.txt', sep=' ', header=None)
        print(len(list(df[0])))

        with open('data/Human/invalid_seq.pkl','rb') as file:
            invalid_seq = pickle.load(file)
        with open('data/Human/invalid_smile.pkl', 'rb') as file:
            invalid_smile = pickle.load(file)

        #Filter out drug molecules or proteins that cannot be graphed in Human.
        for i in invalid_smile:
            index = df[df[0] == i].index.tolist()
            df = df.drop(index)
        print(len(list(df[0])))
        for i in invalid_seq:
            index = df[df[1] == i].index.tolist()
            df = df.drop(index)
        print(len(list(df[0])))

        # Five-fold splitting
        portion = int(0.2 * len(df[0]))
        for fold in range(5):
            if fold < 4:
                df_test = df.iloc[fold * portion:(fold + 1) * portion]
                df_train = pd.concat([df.iloc[:fold * portion], df.iloc[(fold + 1) * portion:]], ignore_index=True)
            if fold == 4:
                df_test = df.iloc[fold * portion:]
                df_train = df.iloc[:fold * portion]
            assert (len(df_test) + len(df_train)) == len(df)
            df_test.to_csv(f'data/Human/test{fold+1}.csv',index=False, header=['compound_iso_smiles','target_sequence','affinity'])
            df_train.to_csv(f'data/Human/train{fold+1}.csv',index=False, header=['compound_iso_smiles','target_sequence','affinity'])
    else:
        print('convert data from DeepDTA for ', dataset)
        # Read the data from the raw file.
        fpath = 'data/' + dataset + '/'
        train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
        train_fold = [ee for e in train_fold for ee in e ]
        valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
        ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
        drugs = []
        prots = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
            drugs.append(lg)
        for t in proteins.keys():
            prots.append(proteins[t])

        if dataset == 'davis':
            affinity = [-np.log10(y/1e9) for y in affinity] # Perform a negative logarithmic transformation on the affinity data in the Davis dataset.

        affinity = np.asarray(affinity)
        opts = ['train','test']
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity)==False)  #Filter out invalid affinity data.
            if opt=='train':
                rows,cols = rows[train_fold], cols[train_fold]
            elif opt=='test':
                rows,cols = rows[valid_fold], cols[valid_fold]
            with open('data/' + dataset + '/' + opt + '.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    f.write(','.join(map(str,ls)) + '\n')
        print('\ndataset:', dataset)
        print('train_fold:', len(train_fold))
        print('test_fold:', len(valid_fold))
        print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))

def construct_graph(args):
    """
    Construct topological graph for protein (pro_data) and drug (mol_data), respectively.
    No value is returned, but the corresponding graph data file is eventually generated.
    :param dataset: dataset name
    :return: None
    """
    print('Construct graph for ', args.dataset)
    ## 1. generate drug graph dict.
    compound_iso_smiles = []
    if args.dataset == 'Human':
        opts = ['train1', 'test1']
    else:
        opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv(f'data/{args.dataset}/' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
    compound_iso_smiles = set(compound_iso_smiles)
    drug_id_mol_tup = []
    symbols = list()
    for smile in compound_iso_smiles:
        mol = Chem.MolFromSmiles(smile.strip())
        if mol is not None:
            drug_id_mol_tup.append((smile, mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())

    symbols = list(set(symbols))
    smile_graph = {smile: generate_drug_data(mol, symbols) for smile, mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
    print('drug graph is constructed successfully!')

    ## 2.generate protein graph dict.
    seq_dict = {}
    with open(f'data/{args.dataset}/{args.dataset}_dict.txt', 'r') as file:
        for line in file.readlines():
            line = line.lstrip('>').strip().split('\t')
            seq_dict[line[1]] = line[0]
    seq_feats = {}
    for pro_id, seq in seq_dict.items():
        seq_feat = seq_to_feat(seq)
        seq_feats[seq] = seq_feat
    print('protein graph is constructed successfully!')

    ## 3. Serialized graph data
    with open(f'data/{args.dataset}/mol_data_M.pkl', 'wb') as smile_file:
        pickle.dump(smile_graph, smile_file)
    with open(f'data/{args.dataset}/pro_data_M.pkl', 'wb') as seq_file:
        pickle.dump(seq_feats, seq_file)


def fold_split_for_davis():
    """ The train set of davis was split into 5 subsets for finetuning hyper-parameter."""
    df = pd.read_csv('data/davis/train.csv')
    portion = int(0.2 * len(df['affinity']))

    for fold in range(5):
        if fold < 4:
            df_test = df.iloc[fold * portion:(fold + 1) * portion]
            df_train = pd.concat([df.iloc[:fold * portion], df.iloc[(fold + 1) * portion:]], ignore_index=True)
        if fold == 4:
            df_test = df.iloc[fold * portion:]
            df_train = df.iloc[:fold * portion]
        assert (len(df_test) + len(df_train)) == len(df)
        df_test.to_csv(f'data/davis/5 fold/test{fold + 1}.csv', index=False)
        df_train.to_csv(f'data/davis/5 fold/train{fold + 1}.csv', index=False)


def main(args):
    data_split(args.dataset)
    construct_graph(args)
    fold_split_for_davis()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'davis', help='dataset name',choices=['davis','kiba','Human'])
    args = parser.parse_args()
    main(args)

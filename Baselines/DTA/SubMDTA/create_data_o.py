import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
from collections import defaultdict



def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


# 使用n-gram 代替原来的编码表示
word_dict_2 = defaultdict(lambda: len(word_dict_2))
word_dict_3 = defaultdict(lambda: len(word_dict_3))
word_dict_4 = defaultdict(lambda: len(word_dict_4))


def split_sequence(sequence, ngram):

    sequence = '-' + sequence + '='

    if ngram == 2:
        words = [word_dict_2[sequence[i:i+ngram]]
                for i in range(len(sequence)-ngram+1)]
    elif ngram == 3:
        words = [word_dict_3[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
    else:
        words = [word_dict_4[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
    # return np.array(words)
    return words





# from DeepDTA data
all_prots = []
datasets = ['davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
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
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train','test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)  
        if opt=='train':
            rows,cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows,cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
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
    all_prots += list(set(prots))
    
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000



compound_iso_smiles = []
for dt_name in ['davis']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g


#
def fun(prots,ngram):
    # XT = [seq_cat(t) for t in train_prots]
    XT = [split_sequence(t, ngram=ngram) for t in prots]


    # tran_prots  ['蛋白质序列'

    seq_max = 1200
    proteins= []
    proteins_mask = []
    for pro in XT:  # 可能有列表和np数组
        pro_len = len(pro)
        if pro_len > seq_max:
            pro = pro[:seq_max]
            mask = [1] * seq_max
        else:
            pro = pro + [0] * (seq_max - pro_len)
            mask = [1] * pro_len + [0] * (seq_max - pro_len)
        proteins.append(pro)
        proteins_mask.append(mask)


    return proteins
    # return XT_2, XT_3, XT_4






datasets = ['davis']
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):

        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])

        # train_proteins, train_mask = fun(train_prots)
        pro_2, pro_3, pro_4 = fun(train_prots,2),fun(train_prots,3),fun(train_prots,4)

        # train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
        train_drugs, train_Y = np.asarray(train_drugs), np.asarray(train_Y)
        train_pro_2, train_pro_3, train_pro_4 = np.asarray(pro_2),np.asarray(pro_3),np.asarray(pro_4)
        
        
        


        # test
        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])

        # XT = [seq_cat(t) for t in test_prots]
        # XT = [split_sequence(t, ngram=3) for t in test_prots]

        t_pro_2, t_pro_3, t_pro_4 = fun(test_prots,2),fun(test_prots,3),fun(test_prots,4)


        # test_proteins, test_mask = fun(test_prots)

        test_drugs, test_Y = np.asarray(test_drugs), np.asarray(test_Y)

        test_pro_2, test_pro_3, test_pro_4 = np.asarray(t_pro_2),np.asarray(t_pro_3),np.asarray(t_pro_4)




        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs,xt_2=train_pro_2, xt_3=train_pro_3,\
                                    xt_4=train_pro_4,y=train_Y,smile_graph=smile_graph)

        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs,xt_2=test_pro_2, xt_3=test_pro_3,\
                                    xt_4=test_pro_4,y=test_Y,smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')


        print('{}{}{}'.format(len(word_dict_2),len(word_dict_3),len(word_dict_4)))

    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')        

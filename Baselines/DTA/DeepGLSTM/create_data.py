# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年04月05日
"""
import json
import os
from operator import index
import torch
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdmolops, SanitizeFlags
import numpy as np
import networkx as nx
import pickle
import pandas as pd
from collections import OrderedDict
import argparse
import torch_geometric.utils as utils
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(filename='pdbbind_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    target_len = 1000
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
    elif dataset == 'PDBbind':
        print('convert data for ', dataset)
        # Read the data from the raw file.
        fpath = 'data/' + dataset + '/'
        alldata_path = f'{fpath}train_test_affinity_data.csv'
        trainset_path = f'{fpath}train.csv'
        testset_path = f'{fpath}test.csv'

        try:
            # 读取原始数据
            df = pd.read_csv(alldata_path)

            # 检查必要列是否存在
            if not all(col in df.columns for col in ['num', 'id', 'affinity']):
                raise ValueError("CSV文件必须包含['num', 'id', 'affinity']列")

            # 随机划分数据集 (4:1比例)
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,  # 20%作为测试集
                random_state=42  # 固定随机种子保证可复现性
            )

            # 重新编号
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df['num'] = range(1, len(train_df) + 1)
            test_df['num'] = range(1, len(test_df) + 1)

            # 保存结果
            train_df.to_csv(trainset_path, index=False, columns=['num', 'id', 'affinity'])
            test_df.to_csv(testset_path, index=False, columns=['num', 'id', 'affinity'])
            # 打印统计信息
            print(f"数据集划分完成：")
            print(f"- 训练集: {len(train_df)} 条 ({len(train_df) / len(df) * 100:.1f}%)")
            print(f"- 测试集: {len(test_df)} 条 ({len(test_df) / len(df) * 100:.1f}%)")
            print(f"文件已保存至:\n  {trainset_path}\n  {testset_path}")

        except FileNotFoundError:
            print(f"错误：文件 {alldata_path} 未找到")
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
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


def self_repair_molecule_file(mol2_path, sdf_path):
    """尝试修复分子文件内容并重新读取"""
    content = None

    # 尝试读取并修复MOL2文件
    if os.path.exists(mol2_path):
        with open(mol2_path, 'r') as f:
            content = f.read()

        # 修复常见问题：替换不合法元素符号
        content = self_repair_content(content)

        # 尝试使用修复后的内容创建分子
        mol = Chem.MolFromMol2Block(content, sanitize=False, removeHs=False)
        if mol is not None:
            return mol

    # 尝试读取并修复SDF文件
    if os.path.exists(sdf_path):
        with open(sdf_path, 'r') as f:
            content = f.read()

        # 修复常见问题：替换不合法元素符号
        content = self_repair_content(content)

        # 尝试使用修复后的内容创建分子
        mol = Chem.MolFromMolBlock(content, sanitize=False, removeHs=False)
        if mol is not None:
            return mol

    return None


def self_repair_content(content):
    """修复分子文件内容中的常见问题"""
    # 替换'Du'为非标准符号（RDKit不识别）
    if 'Du' in content:
        content = content.replace(' Du ', ' U ')  # 假设为铀元素（需确认数据集中的实际意图）
        content = content.replace('Du', 'U')

    # 解决非标准原子类型问题
    replacements = {
        'O.co2': 'O',  # 不合理的原子类型
        'C.ar': 'C',  # 通用化芳香碳
        'N.am': 'N',  # 通用化胺基氮
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    return content

def self_repair_unusual_elements(mol):
    """修复分子中的非标准元素符号"""
    # 获取原子表
    pt = Chem.GetPeriodicTable()

    # 查找并替换非标准原子类型
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()

        # 检查元素是否合法
        try:
            pt.GetAtomicNumber(symbol)
        except:
            logging.warning(f"修复非标准元素: {symbol}")
            # 根据常见情况猜测正确元素
            if symbol == 'Du':
                atom.SetAtomicNum(92)  # 设为铀(uranium)
            else:
                atom.SetAtomicNum(6)  # 默认为碳


def self_partial_sanitization(mol):
    """尝试进行部分sanitization操作"""
    # 尝试修正化合价问题
    Chem.SanitizeMol(mol,
                     sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                 Chem.SanitizeFlags.SANITIZE_ADJUSTHS ^
                                 Chem.SanitizeFlags.SANITIZE_SETAROMATICITY ^
                                 Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    # 尝试手动设置芳环属性（如果sanitize无法Kekulize）
    if mol.HasProp("_NeedsKekulized"):
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_RDKIT)

    # 尝试移除多余氢原子
    try:
        Chem.RemoveHs(mol, sanitize=False)
    except:
        pass  # 忽略无法移除氢的情况

def construct_graph(args):
    """
    Construct topological graph for protein (pro_data) and drug (mol_data), respectively.
    No value is returned, but the corresponding graph data file is eventually generated.
    :param dataset: dataset name
    :return: None
    """
    if args.dataset == 'PDBbind':
        index_path = os.path.join('/media/ST-18T/huasen/PDBbind/PDBbind_v2020/v2020-other-PL')
        fpath = os.path.join('data', args.dataset)
        alldata_path = os.path.join(fpath, 'train_test_affinity_data.csv')

        # df = pd.read_csv(alldata_path)
        # ids = list(df['id'])
        #
        # # 用于收集失败案例的列表
        # failed_cases = []
        # ligand_graph = {}
        #
        # # 准备进度条
        # progress_bar = tqdm(ids, desc='处理PDBbind配体')
        #
        # for id in progress_bar:
        #     mol2_path = os.path.join(index_path, id, f"{id}_ligand.mol2")
        #     sdf_path = os.path.join(index_path, id, f"{id}_ligand.sdf")
        #
        #     # 优先检查文件是否存在
        #     if not os.path.exists(mol2_path) and not os.path.exists(sdf_path):
        #         failed_cases.append((id, "分子文件不存在"))
        #         progress_bar.set_description(f"处理PDBbind配体 (失败: {len(failed_cases)}")
        #         continue
        #
        #     # 增强分子读取过程
        #     mol = None
        #     # 第一尝试：读取MOL2文件（使用不严格的解析）
        #     if os.path.exists(mol2_path):
        #         mol = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
        #
        #     # 第二尝试：读取SDF文件（不净化）
        #     if mol is None and os.path.exists(sdf_path):
        #         suppl = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        #         if suppl is not None:
        #             mol = suppl[0] if len(suppl) > 0 else None
        #
        #     # 第三尝试：如果以上都不成功，尝试修复分子
        #     if mol is None:
        #         try:
        #             # 尝试读取并修复分子文件内容
        #             mol = self_repair_molecule_file(mol2_path, sdf_path)
        #         except Exception as e:
        #             logging.error(f"分子修复失败: {id} - {str(e)}")
        #
        #     # 如果仍然无法读取分子，记录失败
        #     if mol is None:
        #         failed_cases.append((id, "分子解析失败"))
        #         progress_bar.set_description(f"处理PDBbind配体 (失败: {len(failed_cases)}")
        #         continue
        #
        #     # 尝试修复分子问题
        #     try:
        #         # 修复非标准元素问题
        #         self_repair_unusual_elements(mol)
        #
        #         # 尝试部分sanitize操作
        #         self_partial_sanitization(mol)
        #     except Exception as e:
        #         logging.warning(f"分子预处理失败: {id} - {str(e)}")
        #         # 不立即失败，继续尝试生成图数据
        #
        #     # 尝试生成图数据
        #     try:
        #         data = generate_drug_data(mol, None)  # 使用None让函数自动确定符号集
        #         ligand_graph[id] = data
        #     except Exception as e:
        #         logging.error(f"生成图数据失败: {id} - {str(e)}")
        #         failed_cases.append((id, f"图生成失败: {str(e)}"))
        #         progress_bar.set_description(f"处理PDBbind配体 (失败: {len(failed_cases)}")
        #
        # # 输出统计信息
        # success_count = len(ligand_graph)
        # total_count = len(ids)
        # failure_count = len(failed_cases)
        #
        # print(f"\n处理完成: 成功 {success_count}/{total_count}, 失败 {failure_count}")

        ## 2.generate protein graph dict.
        seq_dict = {}
        with open(f'{fpath }/protein_sequences.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(':', 1)  # 只分割第一个冒号
                if len(parts) == 2:
                    protein_id = parts[0]
                    sequence = parts[1]
                seq_dict[protein_id] = sequence
        seq_feats = {}
        for pro_id, seq in seq_dict.items():
            seq_feat = seq_to_feat(seq)
            seq_feats[pro_id] = seq_feat
        print('protein graph is constructed successfully!')

        # ## 3. Serialized graph data
        # with open(f'data/{args.dataset}/mol_data_M.pkl', 'wb') as smile_file:
        #     pickle.dump(ligand_graph, smile_file)
        with open(f'data/{args.dataset}/pro_data_M.pkl', 'wb') as seq_file:
            pickle.dump(seq_feats, seq_file)

    else:
        print('Construct graph for ', args.dataset)
        # ## 1. generate drug graph dict.
        # compound_iso_smiles = []
        # if args.dataset == 'Human':
        #     opts = ['train1', 'test1']
        # else:
        #     opts = ['train', 'test']
        # for opt in opts:
        #     df = pd.read_csv(f'data/{args.dataset}/' + opt + '.csv')
        #     compound_iso_smiles += list(df['compound_iso_smiles'])
        # compound_iso_smiles = set(compound_iso_smiles)
        # drug_id_mol_tup = []
        # symbols = list()
        # for smile in compound_iso_smiles:
        #     mol = Chem.MolFromSmiles(smile.strip())
        #     if mol is not None:
        #         drug_id_mol_tup.append((smile, mol))
        #         symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())
        #
        # symbols = list(set(symbols))
        # smile_graph = {smile: generate_drug_data(mol, symbols) for smile, mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
        # print('drug graph is constructed successfully!')

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

        # ## 3. Serialized graph data
        # with open(f'data/{args.dataset}/mol_data_M.pkl', 'wb') as smile_file:
        #     pickle.dump(smile_graph, smile_file)
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
    # data_split(args.dataset)
    construct_graph(args)
    # fold_split_for_davis()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'kiba', help='dataset name',choices=['davis','kiba','Human','PDBbind'])
    args = parser.parse_args()
    main(args)

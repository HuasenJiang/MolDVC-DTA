# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年04月05日
"""
import warnings
from Bio import BiopythonWarning
warnings.filterwarnings('ignore', category=BiopythonWarning)
import json
import os
import ast
from operator import index
import torch
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
import networkx as nx
import pickle
import pandas as pd
from collections import OrderedDict
import argparse
import torch_geometric.utils as utils
from torch_geometric.data import Data
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
    }

def Affinity_list(path):
    # ----------------所有数据----------------：
    # index_path = 'path/index/INDEX_general_PL_data.2020'
    # # 定义输入文件和输出文件路径
    # index_path = f'{path}/index/INDEX_general_PL_data.2020'
    # output_path = f'data/PDBbind/affinity_data.csv'
    #
    # # 读取原始文件（跳过注释行）
    # with open(index_path, 'r') as f:
    #     lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]
    #
    # # 提取PDB ID和亲和力值
    # data = []
    # for num, line in enumerate(lines, 1):
    #     parts = line.split()
    #     if len(parts) >= 4:  # 确保行有足够列
    #         pdb_id = parts[0]
    #         affinity = parts[3]
    #         data.append([num, pdb_id, affinity])
    #
    # # 创建DataFrame并保存为CSV
    # df = pd.DataFrame(data, columns=['num', 'id', 'affinity'])
    # df.to_csv(output_path, index=False)
    # print(f'Successfully saved to {output_path} with {len(df)} records')

    # ------------训练测试数据-------------：
    # Alldata_path = 'data/PDBbind/affinity_data.csv'
    # train_test_data = 'data/PDBbind/protein_sequences.txt'
    # output_path = 'data/PDBbind/train_test_affinity_data.csv'
    #
    # # 1. 从 protein_sequences.txt 读取所有 ID
    # ids = []
    # with open(train_test_data, 'r') as f:
    #     for line in f:
    #         # 分割每行的 id 和 sequence
    #         parts = line.strip().split(':', 1)  # 只分割第一个冒号
    #         if len(parts) == 2:
    #             protein_id = parts[0]
    #             ids.append(protein_id)
    #
    # # 2. 从 affinity_data.csv 读取并筛选数据
    # df = pd.read_csv(Alldata_path)
    # filtered_df = df[df['id'].isin(ids)].copy()
    #
    # # 3. 重新编号 num 列
    # filtered_df['num'] = range(1, len(filtered_df) + 1)
    #
    # # 4. 保存到新文件
    # filtered_df.to_csv(output_path, index=False, columns=['num', 'id', 'affinity'])
    #
    # print(f"处理完成，共筛选 {len(filtered_df)} 条数据，已保存到 {output_path}")
    #

    # --------检查长度超过1024的样本数：--------------
    train_test_data = 'data/PDBbind/protein_sequences.txt'
    total_samples = 0
    long_sequences = 0
    long_sequence_ids = []

    with open(train_test_data, 'r') as f:
        for line in f:
            # 分割每行的 id 和 sequence
            parts = line.strip().split(':', 1)  # 只分割第一个冒号
            if len(parts) == 2:
                protein_id = parts[0]
                sequence = parts[1]
                total_samples += 1

                # 计算序列长度
                seq_length = len(sequence)

                # 检查是否超过1024
                if seq_length > 1024:
                    long_sequences += 1
                    long_sequence_ids.append((protein_id, seq_length))

    # 输出统计结果
    print(f"总样本数: {total_samples}")
    print(f"长度超过1024的样本数: {long_sequences}")






def get_residues_from_model(model):
    """使用PPBuilder从模型中提取标准氨基酸残基，保持原始顺序"""
    ppb = PPBuilder()
    residues = []

    for chain in model:
        # 构建多肽片段（自动跳过非标准残基）
        peptides = ppb.build_peptides(chain)

        for peptide in peptides:
            # 逐个添加残基，保持它们在肽段中的顺序
            residues.extend(peptide)

    return residues


def parse_pdb_to_seq_cm(pdb_file,threshold=8.0,max_length = 1200):
    """使用PPBuilder提取蛋白质序列"""
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    # 只处理第一个模型
    residues = get_residues_from_model(structure[0])

    sequence = []
    ca_atoms = []
    for residue in residues:
        resname = residue.get_resname()
        aa = aa_codes.get(resname, 'X')

        # 跳过非标准氨基酸
        if aa == 'X':
            continue

        try:
            # 只添加同时有有效氨基酸代码和CA原子的残基
            ca_coord = residue['CA'].get_coord()
            sequence.append(aa)
            ca_atoms.append(ca_coord)
        except KeyError:
            # 跳过缺失CA的残基
            continue

    # 强制截断（防止break未触发时长度超标）
    sequence = sequence[:max_length]
    ca_atoms = ca_atoms[:max_length]

    # 计算contact map（自动与截断后的序列匹配）
    ca_atoms = np.array(ca_atoms)
    dist_matrix = pdist(ca_atoms)
    dist_matrix = squareform(dist_matrix)
    contact_map = (dist_matrix < threshold).astype(int)

    # 最终一致性验证
    assert len(sequence) == contact_map.shape[0] == contact_map.shape[1], \
           f"长度不一致: 序列({len(sequence)}), contact map({contact_map.shape})"

    return ''.join(sequence), contact_map


# def calculate_contact_map(pdb_file, threshold=5.0):
#     """计算蛋白质的contact map，使用相同的残基列表"""
#     parser = PDBParser()
#     structure = parser.get_structure('protein', pdb_file)
#
#     # 获取与序列相同的残基列表
#     residues = get_residues_from_model(structure[0])
#
#     # 提取CA原子坐标
#     ca_atoms = []
#     for residue in residues:
#         try:
#             ca_atoms.append(residue['CA'].get_coord())
#         except KeyError:
#             # 跳过缺失CA的残基
#             continue
#
#     if not ca_atoms:
#         return None
#
#     # 计算contact map
#     ca_atoms = np.array(ca_atoms)
#     dist_matrix = pdist(ca_atoms)
#     dist_matrix = squareform(dist_matrix)
#     contact_map = (dist_matrix < threshold).astype(np.float32)
#     return contact_map


def process_single_pdb(pdb_id, path):
    """处理单个PDB文件，包含详细日志"""
    try:
        logger.info(f"[{pdb_id}] Starting processing")

        # 查找PDB文件
        pdb_dir = os.path.join(path, pdb_id)
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")

        if not os.path.exists(pdb_file):
            error_msg = f"[{pdb_id}] Error: PDB file not found"
            logger.error(error_msg)
            return (pdb_id, None, None, error_msg)

        # 提取序列,contact_map
        sequence,contact_map = parse_pdb_to_seq_cm(pdb_file)
        logger.info(f"[{pdb_id}] Sequence extracted: {sequence[:10]}...")

        # # 计算contact map
        # contact_map = calculate_contact_map(pdb_file)

        if contact_map is not None:
            logger.info(f"[{pdb_id}] Contact map calculated, shape: {contact_map.shape}")
            return (pdb_id, sequence, contact_map, None)
        else:
            error_msg = f"[{pdb_id}] Warning: No CA atoms found in structure"
            logger.warning(error_msg)
            return (pdb_id, None, None, error_msg)

    except Exception as e:
        error_msg = f"[{pdb_id}] Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return (pdb_id, None, None, error_msg)


def protein_seq_contact_map(path):
    """主处理函数，使用并行处理"""
    # 创建输出目录
    os.makedirs('data/PDBbind/contact_map', exist_ok=True)

    # 读取ID列表
    id_affinity_path = 'data/PDBbind/affinity_data.csv'
    df = pd.read_csv(id_affinity_path)
    id_list = df['id'].tolist()

    # 准备并行处理
    num_workers = min(cpu_count(), 8)
    logger.info(f"Starting parallel processing with {num_workers} workers")

    # 使用partial固定path参数
    process_func = partial(process_single_pdb, path=path)

    # 使用进程池处理
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, id_list),
            total=len(id_list),
            desc="Processing PDB files"
        ))

    # 处理结果
    seq_dict = {}
    processed_count = 0
    error_count = 0

    for result in results:
        pdb_id, sequence, contact_map, error_msg = result

        if sequence is not None and contact_map is not None:
            seq_dict[pdb_id] = sequence
            np.save(f'data/PDBbind/contact_map/{pdb_id}.npy', contact_map)
            processed_count += 1
        else:
            error_count += 1

    # 保存序列字典
    with open('data/PDBbind/protein_sequences.txt', 'w') as f:
        for pdb_id, seq in seq_dict.items():
            f.write(f"{pdb_id}:{seq}\n")

    # 打印总结报告
    logger.info("\n=== Processing Summary ===")
    logger.info(f"Total IDs: {len(id_list)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Errors encountered: {error_count}")
    logger.info(f"Success rate: {processed_count / len(id_list) * 100:.2f}%")


def main(data_path):
    Affinity_list(data_path)
    # protein_seq_contact_map(data_path)


if __name__ == '__main__':
    data_path = '/media/ST-18T/huasen/PDBbind/PDBbind_v2020/v2020-other-PL'
    main(data_path)

# class CustomData(Data):
#     '''
#     Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
#     '''
#     def __inc__(self, key, value, *args, **kwargs):
#         if key == 'line_graph_edge_index':
#             return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
#         return super().__inc__(key, value, *args, **kwargs)
#
# def atom_features(atom):
#     results = (one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
#                     one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
#                     one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
#                     one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
#                     [atom.GetIsAromatic()])
#
#     results = np.array(results).astype(np.float32)
#     return results
#
# def one_of_k_encoding(x, allowable_set):
#     if x not in allowable_set:
#         raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
#     return list(map(lambda s: x == s, allowable_set))
#
# def one_of_k_encoding_unk(x, allowable_set):
#     """Maps inputs not in the allowable set to the last element."""
#     if x not in allowable_set:
#         x = allowable_set[-1]
#     return list(map(lambda s: x == s, allowable_set))
#
# def edge_features(bond):
#     '''
#     Get bond features
#     '''
#     bond_type = bond.GetBondType()
#     return torch.tensor([
#         bond_type == Chem.rdchem.BondType.SINGLE,
#         bond_type == Chem.rdchem.BondType.DOUBLE,
#         bond_type == Chem.rdchem.BondType.TRIPLE,
#         bond_type == Chem.rdchem.BondType.AROMATIC,
#         bond.GetIsConjugated(),
#         bond.IsInRing()]).long()
#
# def generate_drug_data(mol_graph, atom_symbols):
#     edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
#     edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (torch.LongTensor([]), torch.FloatTensor([]))
#     edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
#     edge_feats = torch.cat([edge_feats]*2, dim=0) if len(edge_feats) else edge_feats
#     c_size = mol_graph.GetNumAtoms()
#     features = []
#     for atom in mol_graph.GetAtoms():
#         feature = atom_features(atom)
#         features.append(torch.from_numpy( feature / sum(feature)) )
#     features = torch.stack(features)
#
#     # This is the most essential step to convert a node graph to a line graph
#     line_graph_edge_index = torch.LongTensor([])
#     if edge_list.nelement() != 0:
#         conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
#         line_graph_edge_index = conn.nonzero(as_tuple=False).T
#     new_edge_index = edge_list.T
#
#     data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index, edge_attr=edge_feats,c_size = c_size )
#
#     return data
#
# def dic_normalize(dic):
#     """
#     Feature normalization
#     :param dic: A dict for describing residue feature
#     :return: Normalizied feature
#     """
#     max_value = dic[max(dic, key=dic.get)]
#     min_value = dic[min(dic, key=dic.get)]
#     interval = float(max_value) - float(min_value)
#     for key in dic.keys():
#         dic[key] = (dic[key] - min_value) / interval
#     dic['X'] = (max_value + min_value) / 2.0
#     return dic
#
#
# VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
# 				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
# 				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
# 				"U": 19, "T": 20, "W": 21,
# 				"V": 22, "Y": 23, "X": 24,
# 				"Z": 25 }
#
#
#
#
# def seqs2int(target):
#
#     return [VOCAB_PROTEIN[s] for s in target]
#
# def seq_to_feat(seq):
#     """
#     Convert a protein sequence into topo graph (i.e., feature matrix and adjacency matrix )
#     :param pro_id: the UniProt id of protein
#     :param seq: protein sequence
#     :param db: dataset name
#     :return: nodenum, feature matrix, and adjacency matrix(sparse).
#     """
#     target = seqs2int(seq)
#     target_len = 1200
#     if len(target) < target_len:
#         target = np.pad(target, (0, target_len - len(target)))
#     else:
#         target = target[:target_len]
#     target = torch.LongTensor([target])
#     c_size = len(seq)
#     return c_size, target
#
#
#
# def data_split(dataset):
#     """
#     Make dataset spliting and Convert the original data into csv format.
#     No value is returned, but the corresponding csv file is eventually generated.
#     :param dataset: dataset name
#     :return: None
#     """
#     if dataset == 'Human':
#         print('convert human data into 5-fold sub-dataset !')
#         df = pd.read_table('data/Human/Human.txt', sep=' ', header=None)
#         print(len(list(df[0])))
#
#         with open('data/Human/invalid_seq.pkl','rb') as file:
#             invalid_seq = pickle.load(file)
#         with open('data/Human/invalid_smile.pkl', 'rb') as file:
#             invalid_smile = pickle.load(file)
#
#         #Filter out drug molecules or proteins that cannot be graphed in Human.
#         for i in invalid_smile:
#             index = df[df[0] == i].index.tolist()
#             df = df.drop(index)
#         print(len(list(df[0])))
#         for i in invalid_seq:
#             index = df[df[1] == i].index.tolist()
#             df = df.drop(index)
#         print(len(list(df[0])))
#
#         # Five-fold splitting
#         portion = int(0.2 * len(df[0]))
#         for fold in range(5):
#             if fold < 4:
#                 df_test = df.iloc[fold * portion:(fold + 1) * portion]
#                 df_train = pd.concat([df.iloc[:fold * portion], df.iloc[(fold + 1) * portion:]], ignore_index=True)
#             if fold == 4:
#                 df_test = df.iloc[fold * portion:]
#                 df_train = df.iloc[:fold * portion]
#             assert (len(df_test) + len(df_train)) == len(df)
#             df_test.to_csv(f'data/Human/test{fold+1}.csv',index=False, header=['compound_iso_smiles','target_sequence','affinity'])
#             df_train.to_csv(f'data/Human/train{fold+1}.csv',index=False, header=['compound_iso_smiles','target_sequence','affinity'])
#     else:
#         print('convert data from DeepDTA for ', dataset)
#         # Read the data from the raw file.
#         fpath = 'data/' + dataset + '/'
#         train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
#         train_fold = [ee for e in train_fold for ee in e ]
#         valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
#         ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
#         proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
#         affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
#         drugs = []
#         prots = []
#         for d in ligands.keys():
#             lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
#             drugs.append(lg)
#         for t in proteins.keys():
#             prots.append(proteins[t])
#
#         if dataset == 'davis':
#             affinity = [-np.log10(y/1e9) for y in affinity] # Perform a negative logarithmic transformation on the affinity data in the Davis dataset.
#
#         affinity = np.asarray(affinity)
#         opts = ['train','test']
#         for opt in opts:
#             rows, cols = np.where(np.isnan(affinity)==False)  #Filter out invalid affinity data.
#             if opt=='train':
#                 rows,cols = rows[train_fold], cols[train_fold]
#             elif opt=='test':
#                 rows,cols = rows[valid_fold], cols[valid_fold]
#             with open('data/' + dataset + '/' + opt + '.csv', 'w') as f:
#                 f.write('compound_iso_smiles,target_sequence,affinity\n')
#                 for pair_ind in range(len(rows)):
#                     ls = []
#                     ls += [ drugs[rows[pair_ind]]  ]
#                     ls += [ prots[cols[pair_ind]]  ]
#                     ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
#                     f.write(','.join(map(str,ls)) + '\n')
#         print('\ndataset:', dataset)
#         print('train_fold:', len(train_fold))
#         print('test_fold:', len(valid_fold))
#         print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))
#
# def construct_graph(args):
#     """
#     Construct topological graph for protein (pro_data) and drug (mol_data), respectively.
#     No value is returned, but the corresponding graph data file is eventually generated.
#     :param dataset: dataset name
#     :return: None
#     """
#     print('Construct graph for ', args.dataset)
#     ## 1. generate drug graph dict.
#     compound_iso_smiles = []
#     if args.dataset == 'Human':
#         opts = ['train1', 'test1']
#     else:
#         opts = ['train', 'test']
#     for opt in opts:
#         df = pd.read_csv(f'data/{args.dataset}/' + opt + '.csv')
#         compound_iso_smiles += list(df['compound_iso_smiles'])
#     compound_iso_smiles = set(compound_iso_smiles)
#     drug_id_mol_tup = []
#     symbols = list()
#     for smile in compound_iso_smiles:
#         mol = Chem.MolFromSmiles(smile.strip())
#         if mol is not None:
#             drug_id_mol_tup.append((smile, mol))
#             symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())
#
#     symbols = list(set(symbols))
#     smile_graph = {smile: generate_drug_data(mol, symbols) for smile, mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
#     print('drug graph is constructed successfully!')
#
#     ## 2.generate protein graph dict.
#     seq_dict = {}
#     with open(f'data/{args.dataset}/{args.dataset}_dict.txt', 'r') as file:
#         for line in file.readlines():
#             line = line.lstrip('>').strip().split('\t')
#             seq_dict[line[1]] = line[0]
#     seq_feats = {}
#     for pro_id, seq in seq_dict.items():
#         seq_feat = seq_to_feat(seq)
#         seq_feats[seq] = seq_feat
#     print('protein graph is constructed successfully!')
#
#     ## 3. Serialized graph data
#     with open(f'data/{args.dataset}/mol_data_M.pkl', 'wb') as smile_file:
#         pickle.dump(smile_graph, smile_file)
#     with open(f'data/{args.dataset}/pro_data_M.pkl', 'wb') as seq_file:
#         pickle.dump(seq_feats, seq_file)
#
#
# def fold_split_for_davis():
#     """ The train set of davis was split into 5 subsets for finetuning hyper-parameter."""
#     df = pd.read_csv('data/davis/train.csv')
#     portion = int(0.2 * len(df['affinity']))
#
#     for fold in range(5):
#         if fold < 4:
#             df_test = df.iloc[fold * portion:(fold + 1) * portion]
#             df_train = pd.concat([df.iloc[:fold * portion], df.iloc[(fold + 1) * portion:]], ignore_index=True)
#         if fold == 4:
#             df_test = df.iloc[fold * portion:]
#             df_train = df.iloc[:fold * portion]
#         assert (len(df_test) + len(df_train)) == len(df)
#         df_test.to_csv(f'data/davis/5 fold/test{fold + 1}.csv', index=False)
#         df_train.to_csv(f'data/davis/5 fold/train{fold + 1}.csv', index=False)
#
#
# def main(args):
#     data_split(args.dataset)
#     construct_graph(args)
#     fold_split_for_davis()
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default= 'davis', help='dataset name',choices=['davis','kiba','Human'])
#     args = parser.parse_args()
#     main(args)

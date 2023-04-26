"""
Refactors some of kGCN preprocessing functions for speed improvement.
"""
import numpy as np
from rdkit import Chem
from kgcn.data_util import dense_to_sparse, dotdict

def normalize_adj(adj):
    # adj[adj > 0] = 1
    degrees = adj.sum(axis=0)
    degrees[degrees==0] = 1
    adj = adj / np.sqrt(degrees[:,None]) / np.sqrt(degrees)
    adj = dense_to_sparse(adj)
    return adj

def build_data(config, data, return_info):
    # data
    # Num x N x F
    features = data["feature"]
    # Num x (N x N)
    normalize_flag = config["normalize_adj_flag"]

    adj = data["adj"]
    if normalize_flag:
        adj = normalize_adj(adj)
    else:
        adj = dense_to_sparse(adj)
    adj[2][:] = data["max_node_num"]
    all_data = dotdict({})
    all_data.features = features
    all_data.adjs = [[adj]]
    all_data.num = 1

    if not return_info:
        return all_data

    info = dotdict({})
    # features: #graphs x #nodes(graph) x #features
    info.feature_dim = features.shape[2]
    info.graph_node_num = features.shape[1]
    info.feature_enabled = True

    info.sequence_max_length = 0
    info.sequences_vec_dim = 0
    info.adj_channel_num = 1
    info.label_dim = data["label_dim"]
    info.vector_modal_dim = []
    info.vector_modal_name = {}
    return all_data, info

def one_hot(x, allowable_set, use_unknown=False):
    if x not in allowable_set:
        if use_unknown:
            one_hot = np.zeros(len(allowable_set))
            one_hot[-1] = 1
        else:
            raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
    else:
        one_hot = np.zeros(len(allowable_set))
        one_hot[allowable_set.index(x)] = 1
    return one_hot

def atom_features(atom, en_list=None, explicit_H=False, use_sybyl=False, use_electronegativity=False,
                  use_gasteiger=False, degree_dim=17):
    if use_sybyl:
        atom_type = ordkit._sybyl_atom_type(atom)
        atom_list = ['C.ar', 'C.cat', 'C.1', 'C.2', 'C.3', 'N.ar', 'N.am', 'N.pl3', 'N.1', 'N.2', 'N.3', 'N.4', 'O.co2',
                     'O.2', 'O.3', 'S.O', 'S.o2', 'S.2', 'S.3', 'F', 'Si', 'P', 'P3', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                     'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                     'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    else:
        atom_type = atom.GetSymbol()
        atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                     'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                     'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    results = [one_hot(atom_type, atom_list, True),
        one_hot(atom.GetDegree(), list(range(degree_dim))),
        one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6], True),
        np.array([atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]),
        one_hot(atom.GetHybridization(),
                              [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                               Chem.rdchem.HybridizationType.SP3D2], True),
        np.array([atom.GetIsAromatic()])]

    if use_electronegativity:
        results.append(np.array([en_list[atom.GetAtomicNum() - 1]]))
    if use_gasteiger:
        gasteiger = atom.GetDoubleProp('_GasteigerCharge')
        if np.isnan(gasteiger) or np.isinf(gasteiger):
            gasteiger = 0  # because the mean is 0
        results.append(np.array([gasteiger]))

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results.append(one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4], True))
    return np.concatenate(results).astype(np.float32)

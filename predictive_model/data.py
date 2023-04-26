import os
import numpy as np
import json
import argparse
import tensorflow as tf

from rdkit import Chem
from kgcn.preprocessing.utils import atom_features, one_of_k_encoding
from kgcn.data_util import dense_to_sparse, dotdict
from scipy.sparse import coo_matrix
from tqdm import tqdm


def normalize_adj(adj):
    """
    Args:
        adj (list): Adjacency matrix in sparse format
    Returns:
        adj (list): Normalized adjacency matrix in sparse format
    """
    adj = [np.array(a) for a in adj]
    adj = coo_matrix((adj[1], (adj[0][:, 0], adj[0][:, 1])), shape=adj[2])
    degrees = np.squeeze(np.asarray(np.sum(adj, 0)))
    degrees[degrees == 0] = 1
    adj = adj / np.sqrt(degrees[:, None]) / np.sqrt(degrees)
    adj = dense_to_sparse(adj)
    return adj


def bond_features(bond):
    """
    Args:
        bond (Chem.Bond): Chemical bond
    Returns:
        np.array of bond features
    """
    results = one_of_k_encoding(
        bond.GetBondType(),
        allowable_set=[
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
    )
    results += [bond.GetIsConjugated()]
    results += [bond.IsInRing()]
    results += one_of_k_encoding(
        bond.GetStereo(),
        allowable_set=[
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
        ],
    )
    return np.array(results, dtype=np.float32)


def extract_mol_features(mol, atom_num_limit, with_edge_features):
    """
    Args:
        mol (Chem.Mol): RDKit molecule object
        atom_num_limit (int): Maximum number of atom
        with_edge_features (bool): if set to True, edge features will be computed for each node.
            The per node edge features are computed as the sum of the edge features of the edges a node belongs to.
    Returns:
        feature (list): list of list of atom features.
        adj (list): adjacency matrix in sparse format.
        enabled_nodes (int): number of atoms in the molecule.
        edge_feature (list): list of per node edge features.
    """
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
    # Create a adjacency matrix
    mol_adj = Chem.GetAdjacencyMatrix(mol)
    row_num = len(mol_adj)
    adj = np.array(mol_adj, dtype=np.int)
    # Set diagonal elements to 1, fill others with the adjacency matrix from RDkit
    for i in range(row_num):
        adj[i][i] = int(1)
    # Create a feature matrix
    feature = [atom_features(atom, degree_dim=17).tolist() for atom in mol.GetAtoms()]
    for _ in range(atom_num_limit - len(feature)):
        feature.append(np.zeros(len(feature[0]), dtype=np.int).tolist())

    if with_edge_features:
        # Create edge feature matrix
        edge_feature = np.zeros((row_num, row_num, 12))
        for i in range(row_num):
            for j in range(i + 1, row_num):
                if adj[i, j] == 1:
                    bond = mol.GetBondBetweenAtoms(i, j)
                    bond_feat = bond_features(bond)
                    edge_feature[i, j] = bond_feat
                    edge_feature[j, i] = bond_feat

        edge_feature = (
            np.matmul(adj, edge_feature).diagonal().transpose().tolist()
        )  # combines edge features per node based on adjacency matrix
        for _ in range(atom_num_limit - len(edge_feature)):
            edge_feature.append(np.zeros(len(edge_feature[0]), dtype=np.int).tolist())
    else:
        edge_feature = []

    adj = dense_to_sparse(adj)
    adj[2][:] = atom_num_limit
    enabled_nodes = len(mol_adj)

    return feature, adj, enabled_nodes, edge_feature


def build_data(data, return_info, normalize_flag=True, cat_dim=None, max_time=None, ignore_record_prediction=True):
    """
    Args:
        data (dict): dictionnary containing all data
        return_info (bool): if set to True, returns additional outputs
        normalize_flag (bool): if set to True, adjacency matrix are normalized
        cat_dim (Optional, int): if given, sets the number of categories for multiclass task
        max_time (Optional, float): if given, sets the normalization constant used for processing time prediction
        ignore_record_prediction (bool): if set to True, labels related to associated number of records will be ignored
    Returns:
        all_data (dotdict): dictionnary containing processed data
        info (Optional, dotdict): general information about the dataset
    """
    # data
    features = np.asarray(data["features"])
    edge_feats = np.asarray(data["edges"])
    labels = np.asarray(data["labels"])
    time = np.asarray(data["time"])
    max_time = np.max(time) if max_time is None else max_time
    time /= max_time
    steps = np.asarray(data["steps"])
    min_recs = np.asarray(data["min_recs"])
    min_recs = min_recs / np.max(min_recs)
    mean_recs = np.asarray(data["mean_recs"])
    mean_recs = mean_recs / np.max(mean_recs)
    max_recs = np.asarray(data["max_recs"])
    max_recs = max_recs / np.max(max_recs)
    if ignore_record_prediction:
        labels = np.concatenate((labels, steps, time), axis=1)
    else:
        labels = np.concatenate((labels, steps, time, min_recs, mean_recs, max_recs), axis=1)
    enabled_node_nums = np.asarray(data["enabled_nodes"], dtype=np.int32)
    smiles = np.asarray(data["smiles"])

    adjs = data["adjs"]
    if normalize_flag:
        adjs = [[normalize_adj(adj)] for adj in adjs]
    else:
        for i, adj in enumerate(adjs):
            adjs[i] = [[np.array(a) for a in adj]]

    all_data = dotdict({})
    all_data.features = features
    all_data.edge_feats = edge_feats if edge_feats.size else None
    all_data.adjs = adjs
    all_data.num = features.shape[0]
    all_data.labels = labels
    all_data.enabled_node_nums = enabled_node_nums
    all_data.smiles = smiles

    if not return_info:
        return all_data

    info = dotdict({})
    # features: #graphs x #nodes(graph) x #features
    info.feature_dim = features.shape[2]
    info.edge_feats_dim = edge_feats.shape[2] if edge_feats.size else 0
    info.graph_node_num = features.shape[1]
    info.feature_enabled = True
    info.regr_dim = labels.shape[1] - 2
    info.cat_dim = np.max(steps) + 1 if cat_dim is None else cat_dim
    info.max_time = max_time
    info.labels_mapper = {"isSolved": 0, "n_steps": 1, "time": 2}
    info.outputs_mapper = {
        "isSolved": 0,
        "n_steps": [1, 1 + info.cat_dim],
        "time": 1 + info.cat_dim,
    }

    info.sequence_max_length = 0
    info.sequences_vec_dim = 0
    info.adj_channel_num = 1
    info.label_dim = labels.shape[1]
    info.vector_modal_dim = []
    info.vector_modal_name = {}
    return all_data, info


def construct_feed(
    batch_idx,
    placeholders,
    data,
    batch_size=None,
    dropout_rate=0.0,
    is_train=False,
    info=None,
):
    """
    Custom construct feed function
    Args:
        batch_idx (list int): list of indexes to include in batch
        placeholders (list tf.placeholder): list of placeholders to feed
        data (dotdict): dictionnary containing data
        batch_size (int): size of batch
        dropout_rate (float): dropout rate to use by model
        is_train (bool): boolean to indicate training or validation state
        info (dotdict): dictionnary containing info on data
    Returns:
        feed_dict (dict): dictionnary containing inputs for kGCN model
    """
    adjs = data.adjs
    features = data.features
    edge_feats = data.edge_feats
    labels = data.labels
    enabled_node_nums = data.enabled_node_nums

    feed_dict = {}
    if batch_size is None:
        batch_size = len(batch_idx)
    for key, pl in placeholders.items():
        if key == "adjs":
            b_shape = None
            for b, b_pl in enumerate(pl):
                for ch, ab_pl in enumerate(b_pl):
                    if b < len(batch_idx):
                        bb = batch_idx[b]
                        b_shape = adjs[bb][ch][2]
                        val = adjs[bb][ch][1]
                        feed_dict[ab_pl] = tf.SparseTensorValue(
                            adjs[bb][ch][0], val, adjs[bb][ch][2]
                        )
                    else:
                        dummy_idx = np.zeros((0, 2), dtype=np.int32)
                        dummy_val = np.zeros((0,), dtype=np.float32)
                        feed_dict[ab_pl] = tf.SparseTensorValue(
                            dummy_idx, dummy_val, b_shape
                        )
        elif key == "features" and features is not None:
            temp_features = np.zeros(
                (batch_size, features.shape[1], features.shape[2]), dtype=np.float32
            )
            temp_features[: len(batch_idx), :, :] = features[batch_idx, :, :]
            feed_dict[pl] = temp_features
        elif key == "edge_feats":
            if edge_feats is not None:
                temp_edge_feats = np.zeros(
                    (batch_size, edge_feats.shape[1], edge_feats.shape[2]),
                    dtype=np.float32,
                )
                temp_edge_feats[: len(batch_idx), :, :] = edge_feats[batch_idx, :, :]
                feed_dict[pl] = temp_edge_feats
        elif key == "labels":
            if len(labels.shape) == 1:
                labels = labels[:, np.newaxis]
            temp_labels = np.zeros((batch_size, labels.shape[1]), dtype=np.float32)
            temp_labels[: len(batch_idx), :] = labels[batch_idx, :]
            feed_dict[pl] = temp_labels
        elif key == "mask":
            mask = np.zeros((batch_size,), np.float32)
            mask[: len(batch_idx)] = 1
            feed_dict[pl] = mask
        elif key == "dropout_rate":
            feed_dict[pl] = dropout_rate
        elif key == "is_train":
            feed_dict[pl] = is_train
        elif key == "enabled_node_nums" and enabled_node_nums is not None:
            temp_enabled_node_nums = np.zeros((batch_size,), np.int32)
            temp_enabled_node_nums[: len(batch_idx)] = np.squeeze(
                enabled_node_nums[batch_idx]
            )
            feed_dict[pl] = temp_enabled_node_nums
        else:
            print(
                f"Non supported key encountered in while constructing feed dict {key}"
            )
    return feed_dict

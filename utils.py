import copy
from functools import wraps, reduce
import socket
import os
from operator import mul
import sys
from statistics import mean
import time

import numpy as np
from rdkit.Chem import AllChem, RWMol
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction

from kgcn.data_util import dense_to_sparse
from kgcn.preprocessing.utils import atom_features
from model_modules import predict_templates


class MoleculeUtils:
    @staticmethod
    def generate_ecfp(mol, radius=2, bits=2048):
        """ Create Extended Connectivity FingerPrint
        Args:
            mol (Mol Object):
            radius (int):
            bits (int):
        Returns:
            Numpy array type ECFP
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits).ToBitString()
        return np.asarray([[int(i) for i in list(fp)]])

    @staticmethod
    def generate_gcn_descriptor(mol, atom_num_limit, label_dim):
        """ Create GCN descriptor (adj, feat, label)
        Args:
            mol (Mol Object):
            atom_num_limit (int):
            label_dim (int):
        Returns:
            adj, feature, label
        """
        # Prepare dummy label information
        label_data = np.zeros(label_dim)
        label_mask = np.zeros_like(label_data)
        label_mask[~np.isnan(label_data)] = 1
        # for index, mol in enumerate(mol):
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ADJUSTHS)
        # Create a adjacency matrix
        mol_adj = Chem.GetAdjacencyMatrix(mol)
        row_num = len(mol_adj)
        adj = np.array(mol_adj, dtype=np.int)
        # Set diagonal elements to 1, fill others with the adjacency matrix from RDkit
        for i in range(row_num):
            adj[i][i] = int(1)
        # Create a feature matrix
        feature = [atom_features(atom, degree_dim=17) for atom in mol.GetAtoms()]
        for _ in range(atom_num_limit - len(feature)):
            feature.append(np.zeros(len(feature[0]), dtype=np.int))

        adj = dense_to_sparse(adj)
        adj[2][:] = atom_num_limit
        obj = {
            "feature": np.asarray([feature]),
            "adj": np.asarray([adj]),
            "label": np.asarray([label_data]),
            "mask_label": np.asarray([label_mask]),
            "max_node_num": atom_num_limit
        }
        return obj

    @staticmethod
    def update_mol_condition(mol_conditions, mols, divided_mols, start_materials, idx):
        """ Update the molecule condition if the molecules in start materials
        Args:
            mol_conditions (list[int]):
            mols (list[Mol Object]):
            divided_mols (list[Mol Object]):
            start_materials (set[str]):
            idx (int):
        Returns:
            "1" if the molecule is in start materials otherwise "0"
        """
        mols.pop(idx)
        mol_conditions.pop(idx)
        for divided_mol in divided_mols:
            mols.append(divided_mol)
            smiles = Chem.MolToSmiles(divided_mol, canonical=True)
            if SearchUtils.sequential_search(smiles, start_materials):
                mol_conditions.append(1)
            else:
                mol_conditions.append(0)

    @staticmethod
    def get_unsolved_mol_condition_idx(mol_conditions):
        """ Get indexes of mol_conditions whose condition is 0
        Args:
            mol_conditions (list[int]):
        Returns:

        """
        unsolved_idxs = []
        for i in range(len(mol_conditions)):
            if mol_conditions[i] == 0:
                unsolved_idxs.append(i)
        return unsolved_idxs

    @staticmethod
    def is_valid(mol):
        """ Check whether Mol Object is valid
        Args:
            mol (list[Mol Object]):
        Returns:
            True if mol is valid otherwise False
        """
        flag = Chem.SanitizeMol(mol, catchErrors=True)
        return True if flag == Chem.SANITIZE_NONE else False


class ReactionUtils:
    """
    Attributes:
        mol (Mol Object):
    """
    mol = None
    rxn_candidates = []
    sorted_rxn_prob_list = None
    sorted_rxn_prob_idxs = None

    def __init__(self, mol):
        """ A constructor of ReactionUtils
        Args:
            mol (Mol Object):
        """
        self.mol = mol

    @staticmethod
    def react_product_to_reactants(product, rxn_rule, gateway=None):
        """
        Args:
            product (Mol Object):
            rxn_rule (Chemical Reaction):
            gateway (JavaGateway):
        Returns:
            list(molecule object)
        """
        return_list = []
        if gateway:
            product = Chem.MolToSmiles(product)
            try:
                reactants_list = gateway.entry_point.reactProductToReactants(product, rxn_rule)
                for reactants in reactants_list:
                    if reactants is None or None in reactants:
                        continue
                    reactants = [Chem.MolFromSmiles(m) for m in reactants]
                    if reactants and None not in reactants:
                        return_list.append(reactants)
                return return_list if return_list else None
            except:
                return None
        if ChemicalReaction.Validate(rxn_rule)[1] == 1 or rxn_rule.GetNumReactantTemplates() != 1:
            return None
        reactants_list = rxn_rule.RunReactants([product, ])
        if not reactants_list:
            return None
        for reactants in reactants_list:
            for reactant in reactants:
                if not MoleculeUtils.is_valid(reactant):
                    continue
            return_list.append(reactants)
        return return_list if return_list else None

    def set_reaction_candidates_and_probabilities(self, model, rxn_rules, model_name, config):
        """
        Args:
            model: Tensorflow model or Keras model instance
            rxn_rules (list[Chemical Reaction]):
            model_name (str):
            config (dict):
        """
        if config['descriptor'] == 'ECFP':
            input_mol = MoleculeUtils.generate_ecfp(self.mol)
            rxn_prob_list = predict_templates(model, input_mol, model_name, config)
        elif config['descriptor'] == 'GCN':
            input_mol = None
            if model_name == 'expansion':
                input_mol = MoleculeUtils.generate_gcn_descriptor(self.mol, config['max_atom_num'], len(rxn_rules))
            elif model_name == 'rollout':
                input_mol = MoleculeUtils.generate_gcn_descriptor(self.mol, config['max_atom_num'], len(rxn_rules))
            rxn_prob_list = predict_templates(model, input_mol, model_name, config)
        else:
            print("[ERROR] Set 'descriptor' to ECFP or GCN")
            sys.exit(1)
        self.sorted_rxn_prob_idxs = np.argsort(-rxn_prob_list)
        self.sorted_rxn_prob_list = rxn_prob_list[self.sorted_rxn_prob_idxs]
        self.rxn_candidates = self.get_reaction_candidates(rxn_rules, config["expansion_num"])

    @staticmethod
    def get_reactions(rxn_rule_path, save_dir, use_reaction_complement=False):
        def complement_reaction(rxn_template):
            if rxn_template.GetNumProductTemplates() != 1:
                print("[ERROR] A reaction template has only one product template.")
                sys.exit(1)
            pro = rxn_template.GetProductTemplate(0)
            rw_pro = RWMol(pro)
            amaps_pro = {a.GetAtomMapNum() for a in pro.GetAtoms()}
            amaps_rcts = {a.GetAtomMapNum() for rct in rxn_template.GetReactants() for a in rct.GetAtoms()}
            amaps_not_in_rcts = amaps_pro.intersection(amaps_rcts)
            for amap in amaps_not_in_rcts:
                aidx = [a.GetIdx() for a in rw_pro.GetAtoms() if a.GetAtomMapNum() == amap][0]
                rw_pro.RemoveAtom(aidx)
            m = rw_pro.GetMol()
            if '.' in Chem.MolToSmarts(m):
                return
            if (m.GetNumAtoms() == 0) or (m.GetNumAtoms() == 1 and m.GetAtomWithIdx(0).GetSymbol() in {"*", None}):
                return
            rxn_template.AddReactantTemplate(m)

        with open(rxn_rule_path, 'r') as f:
            lines = [l.strip('\n') for l in f.readlines()]
        if use_reaction_complement:
            rxn_templates = []
            for l in lines:
                try:
                    rxn_templates.append(AllChem.ReactionFromSmarts(l))
                except Exception as e:
                    rxn_templates.append(l)

            for rxn_template in rxn_templates:
                if type(rxn_template) == ChemicalReaction:
                    complement_reaction(rxn_template)

            out_reactions = [AllChem.ReactionToSmarts(rt) if type(rt) == ChemicalReaction else rt for rt in rxn_templates]

            basename, ext = os.path.splitext(os.path.basename(rxn_rule_path))
            with open(os.path.join(save_dir, f"{basename}_complemented{ext}"), 'w') as f:
                f.writelines('\n'.join(out_reactions))
            return out_reactions
        else:
            return lines

    @staticmethod
    def get_reverse_reactions(rxn_rule_path):
        """
        Args:
            rxn_rule_path (str):
        Returns:
            list[RxnMolecule]
        """
        with open(rxn_rule_path, 'r') as f:
            lines = f.readlines()
        split_rxn_rules = [l.strip().split('>>') for l in lines]
        reverse_rxn_str = ['>>'.join(split_rxn_rule[::-1]) for split_rxn_rule in split_rxn_rules]
        return [AllChem.ReactionFromSmarts(r) for r in reverse_rxn_str]

    def get_reaction_candidates(self, rxn_rules, expansion_num, top_number=None):
        """
        Args:
            rxn_rules (list[Chemical Reaction]):
            expansion_num (int):
            top_number (int):
        Returns:

        """
        idxs = []
        probs = []
        if top_number is None:  # for expansion
            for i in range(len(self.sorted_rxn_prob_idxs)):
                probs.append(self.sorted_rxn_prob_list[i])
                idxs.append(self.sorted_rxn_prob_idxs[i])
                if i+1 >= expansion_num:
                    break
            rxn_cands = [rxn_rules[i] for i in idxs]
            self.sorted_rxn_prob_list = probs
            return rxn_cands
        else:  # for rollout
            idxs = [self.sorted_rxn_prob_idxs[i] for i in range(top_number)]
            rxn_cands = [rxn_rules[i] for i in idxs]
            return rxn_cands

    @staticmethod
    def predict_reactions(rxn_rules, model, mol, model_name, config, top_number=None):
        """
        Args:
            rxn_rules (list[Chemical Reaction]):
            model: Tensorflow model or Keras model instance
            mol (Molecule):
            model_name (str):
            config (dict):
            top_number (int): if not None, get top-N prediction values
        Returns:
            Lists of predicted Chemical Reaction(s) and reaction probabilities
        """
        rxn = ReactionUtils(mol)
        rxn.set_reaction_candidates_and_probabilities(model, rxn_rules, model_name, config)
        if top_number is None:
            return rxn.get_reaction_candidates(rxn_rules, config["expansion_num"]), rxn.sorted_rxn_prob_list
        else:
            return rxn.get_reaction_candidates(rxn_rules, config["expansion_num"], top_number), rxn.sorted_rxn_prob_list


class SearchUtils:
    @staticmethod
    def sequential_search(mol, start_materials):
        """
        Args:
            mol (str):
            start_materials (set[str]):
        Returns:
            Boolean
        """
        return True if mol in start_materials else False

    @staticmethod
    def is_proved(mol_conditions):
        """
        Args:
            mol_conditions (list[int]):
        Returns:

        """
        return all([i == 1 for i in mol_conditions])

    @staticmethod
    def is_terminal(mols, gateway=None):
        """
        Args:
            mols (list[Mol Object]):
            gateway (JavaGateway):
        Returns:

        """
        str_mols = [Chem.MolToSmiles(m) for m in mols]
        return gateway.entry_point.isTerminal(str_mols)

    @staticmethod
    def is_loop_route(mols, node):
        """ Check whether a molecule is in a route.
        Args:
            mols (list[Mol Object]):
            node (Node):
        Returns:
            True if a molecule is in a route otherwise False
        """
        mols = [Chem.MolToSmiles(m) for m in mols]
        while node is not None:
            unresolved_mols = set(node.state.mols[i] for i, c in enumerate(node.state.mol_conditions) if c == 0)
            unresolved_mols = [Chem.MolToSmiles(m) for m in unresolved_mols]
            for m in mols:
                if m in unresolved_mols:
                    return True
            node = node.parent_node
        return False


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        print("[INFO] start")
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print(f"[INFO] done in {elapsed_time:5f} s")
        return result
    return wrapper


def calculate_cdscore(product, reactants):
    """
    Args:
        product (Mol object):
        reactants (list(Mol object)):
    Returns:
        score (float)
        return 1 if a molecule was divided evenly otherwise 0 <= x < 1.
    """
    if len(reactants) == 1:
        return 0.
    pro_atom_num = product.GetNumAtoms()
    rct_atom_nums = [m.GetNumAtoms() for m in reactants]
    scale_factor = pro_atom_num / len(rct_atom_nums)
    abs_errors = [abs(r - scale_factor) for r in rct_atom_nums]
    return 1 / (1 + mean(abs_errors))


def calculate_asscore(mol_condition_before, mol_condition_after, num_divided_mols):
    """
    Args:
        mol_condition_before (list):
        mol_condition_after (list):
        num_divided_mols (int):
    Returns:
        return 1 if all divided molecules were starting materials otherwise 0 =< x < 1.
    """
    if num_divided_mols == 1:
        return 0.
    return (mol_condition_after.count(1) - mol_condition_before.count(1)) / num_divided_mols


def calculate_rdscore(product, reactants):
    """
    Args:
        product (Mol object):
        reactants (list(Mol object)):
    Returns:
        score (float)
        return 1 if a number of rings in a product is reduced otherwise 0.
    """
    try:
        pro_ring_num = product.GetRingInfo().NumRings()
    except Exception as e:
        product.UpdatePropertyCache()
        Chem.GetSymmSSSR(product)
        pro_ring_num = product.GetRingInfo().NumRings()
    rct_ring_nums = sum([m.GetRingInfo().NumRings() for m in reactants])
    rdscore = pro_ring_num - rct_ring_nums
    return 1. if rdscore > 0 else 0.


def calculate_stscore(reactants, reaction_template):
    """
    Args:
        reactants (list(Mol object)):
        reaction_template (str):
    Returns:
        score (float)
        return 1 if each reactant has a respective substructure in reaction template otherwise 1 / number of the combination.
    """
    patts_for_rct = [Chem.MolFromSmarts(patt) for patt in reaction_template.split(">>")[0].split(".")]
    match_patts = []
    for rct, patt in zip(reactants, patts_for_rct):
        match_patts.append(len(rct.GetSubstructMatches(patt, useChirality=True)))
    match_patts = [1 if patt == 0 else patt for patt in match_patts]
    return 1 / reduce(mul, match_patts)


def is_port_in_used(port):
    """
    Args:
        port (int):
    Returns:
        return True if the port is in used otherwise False
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def get_default_config():
    """
    Returns:
        return config dict
    """
    config = {
        "max_atom_num": 50,
        "search_count": 100,
        "rollout_depth": 5,
        "expansion_model": "model/model.sample.ckpt",
        "expansion_rules": "data/sample_reaction_rule.csv",
        "rollout_model": "model/model.sample.ckpt",
        "rollout_rules": "data/sample_reaction_rule.csv",
        "descriptor": "GCN",
        "gcn_expansion_config": "model/sample.json",
        "gcn_rollout_config": "model/sample.json",
        "starting_material": "data/starting_materials.smi",
        "save_result_dir": "result",
        "target": "data/sample.mol"
    }
    return config


def get_node_info(node, ws):
    """
    Args:
        node (Node):
        ws (list(int)): knowledge weights. [cdscore, rdscore, asscore, stscore]
    Returns:
        return  node information for a searched tree analysis
        node information: self node, parent node, depth, score, RDScore, CDScore, STScore, ASScore
    """
    return (f"{id(node)}\t"
            f"{id(node.parent_node)}\t"
            f"{node.depth}\t"
            f"{node.total_scores / node.visits}\t"
            f"{node.state.rdscore}\t"
            f"{node.state.cdscore * ws[0]}\t"
            f"{node.state.stscore * ws[3]}\t"
            f"{node.state.asscore * ws[2]}")

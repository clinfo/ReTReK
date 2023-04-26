from functools import reduce
import socket
import os
from operator import mul
import sys
from statistics import mean
import time
import json

import numpy as np
import torch
from rdkit.Chem import AllChem, RWMol
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction

from model_modules import predict_templates
from preprocessing_modules import atom_features
from reimplemented_libraries import CxnUtils


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
    def generate_count_ecfp(mol, radius=2, bits=2048):
        """ Create Extended Connectivity Fingerprint with counts
        Args:
            mol (Mol Object):
            radius (int):
            bits (int):
        Returns:
            Numpy array type ECFP with counts
        """
        Chem.SanitizeMol(mol)
        bit_info = {}
        fgp = np.zeros(bits)
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, bits, bitInfo=bit_info)
        for bit_id, active in bit_info.items():
            fgp[bit_id] = len(active)
        return fgp

    @staticmethod
    def generate_reaction_count_ecfp(product, reactants, radius=2, bits=2048, pre_computed_product_fgp=None):
        """ Create Extended Connectivity Fingerprint with counts of reaction
        Args:
            product (Mol Object):
            reactants (List[Mol Object]):
            radius (int):
            bits (int):
        Returns:
            Numpy array type ECFP with counts
        """
        p_fgp = MoleculeUtils.generate_count_ecfp(product, radius, bits) if pre_computed_product_fgp is None else pre_computed_product_fgp
        r_fgp = np.sum([MoleculeUtils.generate_count_ecfp(r, radius, bits) for r in reactants], axis=0)
        return p_fgp - r_fgp

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

        obj = {
            "feature": np.asarray([feature]),
            "adj": adj,
            "label_dim": label_dim,
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
            if "inchi" in start_materials:
                AllChem.SanitizeMol(divided_mol)
                to_check = Chem.MolToInchiKey(divided_mol)
            else:
                to_check = Chem.MolToSmiles(divided_mol, canonical=True)
            if SearchUtils.sequential_search(to_check, start_materials):
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

    def __init__(self, mol=None):
        """ A constructor of ReactionUtils
        Args:
            mol (Mol Object):
        """
        self.mol = mol
        self.predict_reaction_cache = {}
        self.product_to_reactants_cache = {}
        self.rxn_candidates = []
        self.sorted_rxn_prob_list = None
        self.sorted_rxn_prob_idxs = None

    def react_product_to_reactants(self, product, rxn_rule, gateway=None):
        """
        Args:
            product (Mol Object):
            rxn_rule (Chemical Reaction):
            gateway (JavaGateway):
        Returns:
            list(molecule object)
        """
        return_list = []
        if (product, rxn_rule) in self.product_to_reactants_cache:
            return self.product_to_reactants_cache[(product, rxn_rule)]
        if gateway:
            try:
                product_mol = Chem.MolToSmiles(product)
                if isinstance(gateway, CxnUtils):
                    reactants_list = gateway.react_product_to_reactants(product_mol, rxn_rule)
                else:
                    reactants_list = gateway.entry_point.reactProductToReactants(product_mol, rxn_rule)
                for reactants in reactants_list:
                    if reactants is None or None in reactants:
                        continue
                    reactants = [Chem.MolFromSmiles(m) for m in reactants]
                    if reactants and None not in reactants:
                        return_list.append(reactants)
                self.product_to_reactants_cache[(product, rxn_rule)] = return_list if return_list else None
                return self.product_to_reactants_cache[(product, rxn_rule)]
            except:
                self.product_to_reactants_cache[(product, rxn_rule)] = None
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
        canonical_smiles = AllChem.MolToSmiles(self.mol, canonical=True)
        if canonical_smiles in self.predict_reaction_cache:
            sorted_rxn_prob_list, sorted_rxn_prob_idxs = self.predict_reaction_cache[canonical_smiles]
        else:
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
            sorted_rxn_prob_idxs = np.argsort(-rxn_prob_list)[:config["expansion_num"]]
            sorted_rxn_prob_list = rxn_prob_list[sorted_rxn_prob_idxs][:config["expansion_num"]]
            self.predict_reaction_cache[canonical_smiles] = (sorted_rxn_prob_list, sorted_rxn_prob_idxs)
        self.sorted_rxn_prob_idxs = sorted_rxn_prob_idxs
        self.sorted_rxn_prob_list = sorted_rxn_prob_list
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

    def get_reaction_candidates(self, rxn_rules, expansion_num, top_number=None, cum_prob_thresh=0):
        """
        Args:
            rxn_rules (list[Chemical Reaction]):
            expansion_num (int):
            top_number (int):
        Returns:

        """
        idxs = []
        probs = []
        cum_prob = 0
        counter_limit = top_number if top_number is not None else expansion_num
        probs = self.sorted_rxn_prob_list[:counter_limit]
        idxs = self.sorted_rxn_prob_idxs[:counter_limit]
        if cum_prob_thresh:
            cum_probs = np.cumsum(probs)
            pruned = max(1,len(cum_probs[cum_probs<cum_prob_thresh]))
            probs = probs[:pruned]
            idxs = idxs[:pruned]
        rxn_cands = [rxn_rules[i] for i in idxs]
        if top_number is None:
            self.sorted_rxn_prob_list = probs
        return rxn_cands

    def predict_reactions(self, rxn_rules, model, mol, model_name, config, top_number=None):
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
        self.mol = mol
        self.set_reaction_candidates_and_probabilities(model, rxn_rules, model_name, config)
        cum_prob_thresh = config["cum_prob_thresh"] if config["cum_prob_mod"] else 0
        if top_number is None:
            return self.get_reaction_candidates(rxn_rules, config["expansion_num"], cum_prob_thresh=cum_prob_thresh), self.sorted_rxn_prob_list
        else:
            return self.get_reaction_candidates(rxn_rules, config["expansion_num"], top_number, cum_prob_thresh), self.sorted_rxn_prob_list

    def filter_in_scope_reactions(self, in_scope_model, product_mol, reactants_set_list):
        if in_scope_model is None:
            return reactants_set_list
        product_input = torch.log(torch.FloatTensor(MoleculeUtils.generate_count_ecfp(product_mol, radius=2, bits=16384)) + 1)
        product_fgp = MoleculeUtils.generate_count_ecfp(product_mol, radius=2, bits=2048)
        reaction_input = []
        for reactants in reactants_set_list:
            reaction_input.append(torch.FloatTensor(MoleculeUtils.generate_reaction_count_ecfp(None, reactants, radius=2, bits=2048, pre_computed_product_fgp=product_fgp)))
        product_input = product_input.repeat(len(reactants_set_list), 1)
        reaction_input = torch.stack(reaction_input)
        in_scope_probs = in_scope_model(reaction_input, product_input)
        valid_reactants = [r for p, r in zip(in_scope_probs, reactants_set_list) if p > 0.5]
        return valid_reactants


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
        if isinstance(gateway, CxnUtils):
            return gateway.is_terminal(mols)
        else:
            mols = [Chem.MolToSmiles(mol) for mol in mols]
            return gateway.entry_point.isTerminal(mols)

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

def get_num_ring(mol):
    try:
        ring_num = mol.GetRingInfo().NumRings()
    except Exception as e:
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        ring_num = mol.GetRingInfo().NumRings()
    return ring_num

def calculate_rdscore(product, reactants):
    """
    Args:
        product (Mol object):
        reactants (list(Mol object)):
    Returns:
        score (float)
        return 1 if a number of rings in a product is reduced otherwise 0.
    """
    pro_ring_num = get_num_ring(product)
    rct_ring_nums = sum([get_num_ring(m) for m in reactants])
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


def calculate_intermediate_score(mols, intermediates):
    return np.mean([Chem.MolToSmiles(m) in intermediates for m in mols])


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
        "in_scope_model": None,
        "descriptor": "GCN",
        "gcn_expansion_config": "model/sample.json",
        "gcn_rollout_config": "model/sample.json",
        "starting_material": "data/starting_materials.smi",
        "intermediate_material": None,
        "template_scores": None,
        "save_result_dir": "result",
        "target": "data/sample.mol"
    }
    return config

def get_config(args):
    # Setup config: Arguments take priority over config file
    config = get_default_config()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    config['max_atom_num'] = int(args.max_atom_num or config['max_atom_num'])
    config['search_count'] = int(args.search_count or config['search_count'])
    config['rollout_depth'] = int(args.rollout_depth or config['rollout_depth'])
    config['expansion_model'] = args.expansion_model or config['expansion_model']
    config['in_scope_model'] = args.in_scope_model or config['in_scope_model']
    config['expansion_rules'] = args.expansion_rules or config['expansion_rules']
    config['rollout_model'] = args.rollout_model or config['rollout_model']
    config['rollout_rules'] = args.rollout_rules or config['rollout_rules']
    config['descriptor'] = args.descriptor or config['descriptor']
    config['gcn_expansion_config'] = args.gcn_expansion_config or config['gcn_expansion_config']
    config['gcn_rollout_config'] = args.gcn_rollout_config or config['gcn_rollout_config']
    config['starting_material'] = args.starting_material or config['starting_material']
    config['intermediate_material'] = args.intermediate_material or config['intermediate_material']
    config['template_scores'] = args.template_scores or config["template_scores"]
    config['save_result_dir'] = args.save_result_dir or config['save_result_dir']
    config['target'] = args.target or config['target']
    config['knowledge'] = set(args.knowledge)
    config["knowledge_weights"] = args.knowledge_weights
    config['save_tree'] = args.save_tree
    config['selection_constant'] = args.sel_const
    config['expansion_num'] = args.expansion_num
    config['cum_prob_mod'] = args.cum_prob_mod
    config['cum_prob_thresh'] = args.cum_prob_thresh
    return config

def get_node_info(node, ws):
    """
    Args:
        node (Node):
        ws (list(int)): knowledge weights. [cdscore, rdscore, asscore, stscore, intermediate_score, template_score]
    Returns:
        return  node information for a searched tree analysis
        node information: self node, parent node, depth, score, RDScore, CDScore, STScore, ASScore, IntermediateScore, TemplateScore
    """
    return (f"{id(node)}\t"
            f"{id(node.parent_node)}\t"
            f"{node.depth}\t"
            f"{node.total_scores / node.visits}\t"
            f"{node.state.rdscore}\t"
            f"{node.state.cdscore * ws[0]}\t"
            f"{node.state.stscore * ws[3]}\t"
            f"{node.state.asscore * ws[2]}\t"
            f"{node.state.intermediate_score * ws[4]}\t"
            f"{node.state.template_score * ws[5]}\t")

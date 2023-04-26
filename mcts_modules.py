import copy
import math
import os
import numpy as np
from pathlib import Path
import random
from statistics import mean
from rdkit import Chem

from utils import MoleculeUtils, ReactionUtils, SearchUtils, get_node_info
from visualization import create_images, create_html_file
from reimplemented_libraries import CxnUtils


class State:
    """ State
    Attributes
        mols (list[Mol Object]): RDKit Mol Object
        rxn_rule (Chem Reaction): RDKit Chemical Reaction
        mol_conditions (list[int]): A condition of molecules. "1" if a molecule is in building blocks "0" otherwise
        rxn_applied_mol_idx (int): The index of a reaction-applied molecule in mols
    """
    def __init__(self, mols, rxn_rule=None, mol_conditions=None, rxn_applied_mol_idx=None, stscore=0,
                 cdscore=0, rdscore=0, asscore=0, intermediate_score=0, template_score=0, knowledge="all", knowledge_weights=[1,1,1,1,1,1]):
        """ A constructor of State
        Args:
            mols (list[Mol Object]): RDKit Mol Object
            rxn_rule (Chem Reaction): RDKit Chemical Reaction
            mol_conditions (list[int]): A condition of molecules. "1" if a molecule is in building blocks "0" otherwise
            rxn_applied_mol_idx (int): The index of a reaction-applied molecule in mols
        """
        self.mols = mols
        self.rxn_rule = rxn_rule
        self.mol_conditions = [0] if mol_conditions is None else mol_conditions
        self.rxn_applied_mol_idx = None if rxn_applied_mol_idx is None else rxn_applied_mol_idx
        self.stscore = stscore
        self.cdscore = cdscore
        self.rdscore = rdscore
        self.asscore = asscore
        self.intermediate_score = intermediate_score
        self.template_score = template_score
        knowledge_score = []
        if "cdscore" in knowledge or "all" in knowledge:
            knowledge_score.append(knowledge_weights[0] * self.cdscore)
        if "rdscore" in knowledge or "all" in knowledge:
            knowledge_score.append(knowledge_weights[1] * self.rdscore)
        if "asscore" in knowledge or "all" in knowledge:
            knowledge_score.append(knowledge_weights[2] * self.asscore)
        if "stscore" in knowledge or "all" in knowledge:
            knowledge_score.append(knowledge_weights[3] * self.stscore)
        if "intermediate_score" in knowledge or "all" in knowledge:
            knowledge_score.append(knowledge_weights[4] * self.intermediate_score)
        if "template_score" in knowledge or "all" in knowledge:
            knowledge_score.append(knowledge_weights[5] * self.template_score)
        self.knowledge_score = np.mean(knowledge_score) if knowledge_score else 0

class Node:
    """ Node
    Attributes:
        state (State):
        parent_node (Node):
        child_nodes (list[]):
        node_probs (list[]): Probability of selected reaction rule in the Node
        depth (int):  A depth of Node
        rxn_probs (float):
        total_scores (float):
        visits (int):
        max_length (int):
    """
    def __init__(self, state, parent_node=None, has_child=False, depth=None):
        """ A constructor of Node
        Args:
            state (State):
            parent_node (Node):
            has_child (Boolean): True if the Node has child Node or False otherwise
            depth (int): A depth of the Node
        """
        self.state = state
        self.parent_node = parent_node
        self.child_nodes = []
        self.has_child = has_child
        self.node_probs = []
        self.depth = 0 if depth is None else depth
        self.rxn_probs = 0.
        self.total_scores = 0.
        self.visits = 1
        self.max_length = 10

    def get_best_leaf(self, logger):
        """
        Method to find best leaf of node based on scores/visits solely.
        Args:
            logger (logging.Logger): Logger
        Returns: The leaf Node with max total_scores/visits
        """
        tmp_node = self
        while tmp_node.has_child:
            tmp_node = tmp_node.select_node(0, logger)
        return tmp_node


    def ucb(self, constant):
        """
        Computes UCB for all child nodes.
        Args:
            constant (float): constant to use for UCB computation
        Returns: ucb for all child nodes
        """
        parent_visits = self.visits
        child_visits = np.array([node.visits for node in self.child_nodes])
        probs = np.array([node.node_probs[0] for node in self.child_nodes])
        knowledge_scores = np.array([node.state.knowledge_score for node in self.child_nodes])
        total_scores = np.array([node.total_scores for node in self.child_nodes])
        exploit = total_scores / child_visits
        explore = probs * math.sqrt(parent_visits) / (1 + child_visits)
        ucb = exploit + constant * explore + knowledge_scores
        return ucb


    def select_node(self, constant, logger):
        """ Selection implementation of MCTS
        Define Q(st, a) to total_scores, N(st, a) to child_visits, N(st-1, a) to parent_visits and P(st, a) to p.
        p is a prior probability received from the expansion.

        Args:
            constant (int):
            logger (logging.Logger): Logger
            knowledge (set(str)):
            ws (list(int)): knowledge weights. [cdscore, rdscore, asscore, stscore, intermediate_score, template_score]
        Returns: The Node which has max ucb score
        """
        ucb = self.ucb(constant)
        max_index = random.choice(np.where(ucb==ucb.max())[0])
        node_num = len(self.child_nodes)
        logger.debug(f"\n################ SELECTION ################\n"
                     f"ucb_list:\n {ucb}\n"
                     f"visit: \n{[self.child_nodes[i].visits for i in range(node_num)]}\n"
                     f"child total scores: \n{[self.child_nodes[i].total_scores for i in range(node_num)]}\n"
                     f"parent visits: {self.visits}\n"
                     f"child node probs: \n{[self.child_nodes[i].node_probs for i in range(node_num)]}\n"
                     f"############################################\n")
        return self.child_nodes[max_index]

    def add_node(self, st, new_node_prob):
        """ Add Node as child node to self.
        Args
            st (State):
            new_node_prob (float):
        Returns:
            The child Node which was added to the parent Node
        """
        new_node = Node(st, parent_node=self, depth=self.depth+1)
        new_node.node_probs.append(new_node_prob)
        for p in self.node_probs:
            new_node.node_probs.append(copy.deepcopy(p))
        self.child_nodes.append(new_node)
        if not self.has_child:
            self.has_child = True
        return new_node

    def rollout(self, reaction_util, rxn_rules, rollout_model, start_materials, config, max_atom_num, gateway=None):
        """ Rollout implementation of MCTS
        Args:
            rxn_rules (list[Chemical Reaction]):
            rollout_model: Tensorflow model or Keras model instance
            start_materials (set[str]):
            config (dict):
            max_atom_num (int):
            gateway (JavaGateway):
        Returns:
            A float type rollout score
        """
        mol_cond = copy.deepcopy(self.state.mol_conditions)
        mols = copy.deepcopy(self.state.mols)
        rand_pred_rxns = []

        # Before starting rollout, the state is first checked for being terminal or proved
        unsolved_mols = [mols[i] for i in MoleculeUtils.get_unsolved_mol_condition_idx(mol_cond)]
        if SearchUtils.is_proved(mol_cond):
            return 10.0
        elif SearchUtils.is_terminal(unsolved_mols, gateway=gateway):
            return -1.0
        else:
            for d in range(config['rollout_depth']):
                rand_pred_rxns.clear()
                unsolved_indices = MoleculeUtils.get_unsolved_mol_condition_idx(mol_cond)
                # Random pick a molecule from the unsolved molecules
                unsolved_idx = random.choice(unsolved_indices)
                rand_mol = mols[unsolved_idx]
                if rand_mol.GetNumAtoms() > max_atom_num:
                    return 0.
                # Get top 10 reaction candidate from rand_mol
                rand_pred_rxns, self.rxn_probs = reaction_util.predict_reactions(rxn_rules, rollout_model, rand_mol,
                                                                                 'rollout', config, top_number=10)
                # Random pick a reaction from the reaction candidate
                rand_rxn_cand = random.choice(rand_pred_rxns)
                #
                divided_mols_list = reaction_util.react_product_to_reactants(rand_mol, rand_rxn_cand, gateway=gateway)
                if not divided_mols_list:
                    continue
                if isinstance(gateway, CxnUtils):
                    MoleculeUtils.update_mol_condition(mol_cond, mols, random.choice(divided_mols_list), start_materials, unsolved_idx)
                else:
                    MoleculeUtils.update_mol_condition(mol_cond, mols, divided_mols_list[0], start_materials, unsolved_idx)
                if SearchUtils.is_proved(mol_cond):
                    break
            return mol_cond.count(1) / len(mol_cond)

    def update(self, score):
        """ Update implementation of MCTS
        Args:
            score (float):
        """
        k = 0.99
        self.visits += 1  # the frequency of visits to the State

        prob = sum(self.node_probs)
        length_factor = self.depth - prob
        weight = max(.0, (self.max_length - length_factor) / self.max_length)
        q_score = score * weight
        self.total_scores += q_score


def back_propagation(node, score):
    """
    Args:
        node (Node):
        score (float):
    """
    while node is not None:
        node.update(score)
        node = node.parent_node


def save_route(nodes, save_dir, is_proven, ws):
    """ Save the searched reaction route.
    Args:
        nodes (list[Node]): List of reaction route nodes.
        save_dir (str):
        is_proven (Boolean): Reaction route search has done or not.
        ws (list(int)): knowledge weights. [cdscore, rdscore, asscore, stscore, intermediate_score, template_score]
    """
    is_proven = "proven" if is_proven else "not_proven"
    Path(os.path.join(save_dir, is_proven)).touch()

    mols_nodes = [".".join([Chem.MolToSmiles(mol) for mol in node.state.mols]) for node in nodes]
    #
    state_save_path = os.path.join(save_dir, "state.sma")
    with open(state_save_path, 'w') as f:
        f.write("\n".join(mols_nodes))
    #
    reaction_save_path = os.path.join(save_dir, "reaction.sma")
    rxns = [node.state.rxn_rule for node in nodes if node.state.rxn_rule is not None]
    with open(reaction_save_path, 'w') as f:
        f.write("\n".join(rxns))
    #
    tree_save_path = os.path.join(save_dir, "best_tree_info.csv")
    tree_info = ["self node\t"
                 "parent node\t"
                 "depth\t"
                 "score\t"
                 "RDScore\t"
                 "CDScore\t"
                 "ASScore\t"
                 "IntermediateScore\t"
                 "TemplateScore"]
    tree_info.extend([get_node_info(node, ws) for node in nodes])
    with open(tree_save_path, 'w') as f:
        f.write("\n".join(tree_info))
    # create_images(save_dir, mols_nodes, rxns)
    # create_html_file(save_dir, len(mols_nodes), len(rxns), f"{name_stem}.html")


def print_route(nodes, is_proven, logger):
    """ Print the searched route
    Args:
        nodes (list[Node]): List of reaction route nodes.
        is_proven (Boolean): Reaction route search has done or not.
        logger (logging.Logger): Logger
    """

    message = "Reaction route search done." if is_proven else "[INFO] Can't find any route..."
    logger.info(message)
    route_summary = ""
    route_summary += "\n\n################### Starting Material(s) ###################"
    rxn_rule = None
    idx = -1
    for node in nodes:
        route_summary += (f"\n------ Visit frequency to node: {node.visits} --------\n"
                          f"The total score: {node.total_scores / node.visits}\n"
                          f"The node depth: {node.depth}\n")
        if rxn_rule is not None:
            route_summary += f'[INFO] Apply reverse reaction rule: {rxn_rule}\n'
        rxn_rule = node.state.rxn_rule
        if idx != -1:
            route_summary += f"[INFO] Reaction applied molecule index: {idx}\n"
        idx = node.state.rxn_applied_mol_idx
        for i in range(len(node.state.mols)):
            route_summary += f"{i}: {Chem.MolToSmiles(node.state.mols[i])}\n"
    route_summary += "###################### Target Molecule #####################\n"
    logger.info(route_summary)

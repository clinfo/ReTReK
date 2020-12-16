import copy
import os

from rdkit import Chem
import tqdm

from utils import MoleculeUtils, ReactionUtils, SearchUtils, calculate_cdscore, calculate_rdscore, \
    get_node_info, calculate_asscore, calculate_stscore
from mcts_modules import Node, State, back_propagation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Mcts:
    """ Monte Carlo Tree Search
    Attributes:
        TARGET_MOL (Mol Object): RDKit Mol Object
        EXPANSION_RULES (list[Chem Reaction]): A list of RDKit Chemical Reaction
        ROLLOUT_RULES (list[Chem Reaction]): A list of RDKit Chemical Reaction
        START_MATERIALS (set[str]): A set of building blocks
        root_node (Node): The Node instance of TARGET_MOL
    """
    def __init__(self, target_mol, expansion_rules, rollout_rules, start_materials, config):
        """ A constructor of MCTS
        Args:
            target_mol (Mol Object): RDKit Mol Object
            expansion_rules (list[Chem Reaction]): A list of RDKit Chemical Reaction
            rollout_rules (list[Chem Reaction]): A list of RDKit Chemical Reaction
            start_materials (set[str]): A list of building blocks
            config (dict): configuration file
        """
        self.TARGET_MOL = target_mol
        self.ATOM_NUM_LIMIT = config["max_atom_num"]
        self.EXPANSION_RULES = expansion_rules
        self.ROLLOUT_RULES = rollout_rules
        self.START_MATERIALS = start_materials
        self.root_node = Node(State([self.TARGET_MOL]))
        self.CONFIG = config

    def search(self, expansion_model, rollout_model, logger, gateway=None):
        """ Implementation of Monte Carlo Tree Search
        Args:
            expansion_model: Tensorflow model or Keras model instance
            rollout_model: Tensorflow model or Keras model instance
            logger (logging.Logger): Logger
            gateway (JavaGateway):
        Returns:
            Node class and True if a reaction route is found or Node class and False otherwise
        """
        header = ("self node\t"
                  "parent node\t"
                  "depth\t"
                  "score\t"
                  "RDScore\t"
                  "CDScore\t"
                  "STScore\t"
                  "ASScore")
        tree_info = [header] if self.CONFIG['save_tree'] else None
        for c in tqdm.tqdm(range(self.CONFIG['search_count'])):
            if self.root_node.visits != 0:
                logger.debug(f'Count: {c} Root: visits: {self.root_node.visits} '
                             f'Total scores: {self.root_node.total_scores / self.root_node.visits}')
            # Selection
            tmp_node = self.root_node
            while tmp_node.has_child:
               tmp_node = tmp_node.select_node(self.CONFIG["selection_constant"], logger, self.CONFIG['knowledge'],
                                               self.CONFIG["knowledge_weights"])
            # Expansion
            unsolved_first_idx = tmp_node.state.mol_conditions.index(0)
            first_unsolved_mol_in_tmp_node = tmp_node.state.mols[unsolved_first_idx]
            if first_unsolved_mol_in_tmp_node.GetNumAtoms() > self.ATOM_NUM_LIMIT:
                back_propagation(tmp_node, -1)
                continue
            new_rxn_rules, tmp_node.rxn_probs = ReactionUtils.predict_reactions(
                self.EXPANSION_RULES, expansion_model, first_unsolved_mol_in_tmp_node, 'expansion', self.CONFIG
            )
            for i in range(len(new_rxn_rules)):
                divided_mols_list = ReactionUtils.react_product_to_reactants(
                    first_unsolved_mol_in_tmp_node, new_rxn_rules[i], gateway=gateway)
                if not divided_mols_list:
                    score = -1.0 / len(new_rxn_rules)
                    back_propagation(tmp_node, score)
                    continue
                for divided_mols in divided_mols_list:
                    stscore = calculate_stscore(divided_mols, new_rxn_rules[i])
                    if SearchUtils.is_loop_route(divided_mols, tmp_node):
                        continue
                    new_mols = copy.deepcopy(tmp_node.state.mols)
                    new_mol_conditions = copy.deepcopy(tmp_node.state.mol_conditions)
                    cdscore = calculate_cdscore(first_unsolved_mol_in_tmp_node, divided_mols)
                    rdscore = calculate_rdscore(first_unsolved_mol_in_tmp_node, divided_mols)
                    logger.debug(f"A depth of new node: {tmp_node.depth}\n")
                    logger.debug(f"Reaction template: {new_rxn_rules[i]}")
                    logger.debug(f'Before mol condition: {new_mol_conditions}')
                    logger.debug([Chem.MolToSmiles(m) for m in new_mols])
                    MoleculeUtils.update_mol_condition(
                        new_mol_conditions, new_mols, divided_mols, self.START_MATERIALS, unsolved_first_idx
                    )
                    logger.debug(f'After mol condition: {new_mol_conditions}')
                    logger.debug([Chem.MolToSmiles(m) for m in new_mols])
                    asscore = calculate_asscore(
                        tmp_node.state.mol_conditions, new_mol_conditions, len(divided_mols)
                    )
                    new_state = State(
                        new_mols, new_rxn_rules[i], new_mol_conditions, unsolved_first_idx, stscore,
                        cdscore, rdscore, asscore
                    )
                    leaf_node = tmp_node.add_node(new_state, tmp_node.rxn_probs[i], tmp_node, tmp_node.depth + 1)
                    if SearchUtils.is_proved(new_mol_conditions):
                        back_propagation(leaf_node, 10.)
                        if self.CONFIG["save_tree"]:
                            tree_info.append(get_node_info(leaf_node, self.CONFIG["knowledge_weights"]))
                            with open(os.path.join(self.CONFIG["save_result_dir"], "tree_log.csv"), 'w') as f:
                                f.write('\n'.join(tree_info))
                        return leaf_node, True
                # Rollout
                    score = leaf_node.rollout(
                        self.ROLLOUT_RULES, rollout_model, self.START_MATERIALS, self.CONFIG, self.ATOM_NUM_LIMIT,
                        gateway=gateway
                    )
                # Back propagation
                    back_propagation(leaf_node, score)
                    if self.CONFIG['save_tree']:
                        tree_info.append(get_node_info(leaf_node, self.CONFIG["knowledge_weights"]))
        if self.CONFIG['save_tree']:
            with open(os.path.join(self.CONFIG["save_result_dir"], "tree_log.csv"), 'w') as f:
                f.write('\n'.join(tree_info))
        # for returning the leaf node of the current best route
        tmp_node = self.root_node
        while tmp_node.has_child:
            tmp_node = tmp_node.select_highest_score_node()
        return tmp_node, False



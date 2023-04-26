import copy
import os
import time

from rdkit import Chem
import tqdm

from utils import MoleculeUtils, ReactionUtils, SearchUtils, calculate_cdscore, calculate_rdscore, \
    get_node_info, calculate_asscore, calculate_stscore, calculate_intermediate_score
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
    def __init__(self, target_mol, expansion_rules, rollout_rules, start_materials, intermediate_materials, template_scores, config):
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
        self.INTERMEDIATE_MATERIALS = intermediate_materials
        self.root_node = Node(State([self.TARGET_MOL]))
        self.CONFIG = config
        self.TEMPLATE_SCORES = template_scores

    def _is_necessary_to_compute(self, knowledge_score):
        weight_index = {
            "cdscore": 0,
            "rdscore": 1,
            "asscore": 2,
            "stscore": 3,
            "intermediate_score": 4,
            "template_score": 5
        }[knowledge_score]
        return (knowledge_score in self.CONFIG["knowledge"] or "all" in self.CONFIG["knowledge"]) and (self.CONFIG["knowledge_weights"][weight_index] > 0)

    def search(self, expansion_model, rollout_model, in_scope_model, logger, gateway=None, time_limit=0):
        """ Implementation of Monte Carlo Tree Search
        Args:
            expansion_model: Tensorflow model or Keras model instance
            rollout_model: Tensorflow model or Keras model instance
            in_scope_model: PyTorch model
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
                  "ASScore\t"
                  "IntermediateScore\t"
                  "TemplateScore")
        tree_info = [header] if self.CONFIG['save_tree'] else None
        start = time.time()
        reaction_util = ReactionUtils()
        for c in tqdm.tqdm(range(self.CONFIG['search_count'])):
            if time_limit and time.time() - start > time_limit:
                break
            if self.root_node.visits != 0:
                logger.debug(f'Count: {c} Root: visits: {self.root_node.visits} '
                             f'Total scores: {self.root_node.total_scores / self.root_node.visits}')
            # Selection
            tmp_node = self.root_node
            while tmp_node.has_child:
               tmp_node = tmp_node.select_node(self.CONFIG["selection_constant"], logger)
            # Expansion
            for mol, mol_cond, mol_idx in zip(tmp_node.state.mols, tmp_node.state.mol_conditions, range(len(tmp_node.state.mols))):
                if mol_cond == 1:
                    continue
                if mol.GetNumAtoms() > self.ATOM_NUM_LIMIT:
                    back_propagation(tmp_node, -1)
                    break
                new_rxn_rules, tmp_node.rxn_probs = reaction_util.predict_reactions(
                    self.EXPANSION_RULES, expansion_model, mol, 'expansion', self.CONFIG
                )
                for i in range(len(new_rxn_rules)):
                    divided_mols_list = reaction_util.react_product_to_reactants(
                        mol, new_rxn_rules[i], gateway=gateway)
                    if not divided_mols_list:
                        score = -1.0 / len(new_rxn_rules)
                        back_propagation(tmp_node, score)
                        continue
                    divided_mols_list = reaction_util.filter_in_scope_reactions(in_scope_model, mol, divided_mols_list)
                    for divided_mols in divided_mols_list:
                        if SearchUtils.is_loop_route(divided_mols, tmp_node):
                            continue
                        new_mols = copy.deepcopy(tmp_node.state.mols)
                        new_mol_conditions = copy.deepcopy(tmp_node.state.mol_conditions)
                        logger.debug(f"A depth of new node: {tmp_node.depth}\n")
                        logger.debug(f"Reaction template: {new_rxn_rules[i]}")
                        logger.debug(f'Before mol condition: {new_mol_conditions}')
                        logger.debug([Chem.MolToSmiles(m) for m in new_mols])
                        MoleculeUtils.update_mol_condition(
                            new_mol_conditions, new_mols, divided_mols, self.START_MATERIALS, mol_idx
                        )
                        logger.debug(f'After mol condition: {new_mol_conditions}')
                        logger.debug([Chem.MolToSmiles(m) for m in new_mols])
                        # Computing knowledge scores
                        cdscore, rdscore, stscore, asscore, intermediate_score, template_score = 0, 0, 0, 0, 0, 0
                        if self._is_necessary_to_compute("cdscore"):
                            cdscore = calculate_cdscore(mol, divided_mols)
                        if self._is_necessary_to_compute("rdscore"):
                            rdscore = calculate_rdscore(mol, divided_mols)
                        if self._is_necessary_to_compute("stscore"):
                            stscore = calculate_stscore(divided_mols, new_rxn_rules[i])
                        if self._is_necessary_to_compute("asscore"):
                            asscore = calculate_asscore(
                                tmp_node.state.mol_conditions, new_mol_conditions, len(divided_mols)
                            )
                        if self._is_necessary_to_compute("intermediate_score"):
                            intermediate_score = calculate_intermediate_score(
                                new_mols, self.INTERMEDIATE_MATERIALS
                            )
                        if self._is_necessary_to_compute("template_score"):
                            template_score = self.TEMPLATE_SCORES.get(new_rxn_rules[i], 0)
                        new_state = State(
                            new_mols, new_rxn_rules[i], new_mol_conditions, mol_idx, stscore,
                            cdscore, rdscore, asscore, intermediate_score, template_score, self.CONFIG['knowledge'], self.CONFIG["knowledge_weights"]
                        )
                        leaf_node = tmp_node.add_node(new_state, tmp_node.rxn_probs[i])
                        if SearchUtils.is_proved(new_mol_conditions):
                            back_propagation(leaf_node, 10.)
                            if self.CONFIG["save_tree"]:
                                tree_info.append(get_node_info(leaf_node, self.CONFIG["knowledge_weights"]))
                                with open(os.path.join(self.CONFIG["save_result_dir"], "tree_log.csv"), 'w') as f:
                                    f.write('\n'.join(tree_info))
                            return leaf_node, True
            if tmp_node.has_child:
                # Select most promising leaf node
                leaf_node = tmp_node.select_node(self.CONFIG["selection_constant"], logger)
                # Rollout
                score = leaf_node.rollout(
                    reaction_util, self.ROLLOUT_RULES, rollout_model, self.START_MATERIALS, self.CONFIG, self.ATOM_NUM_LIMIT,
                    gateway=gateway
                    )
                # Back propagation
                back_propagation(leaf_node, score)
                if self.CONFIG['save_tree']:
                    tree_info.append(get_node_info(leaf_node, self.CONFIG["knowledge_weights"]))
            else:
                back_propagation(tmp_node, -1)
        if self.CONFIG['save_tree']:
            with open(os.path.join(self.CONFIG["save_result_dir"], "tree_log.csv"), 'w') as f:
                f.write('\n'.join(tree_info))
        # for returning the leaf node of the current best route
        leaf_node = self.root_node.get_best_leaf(logger)
        return leaf_node, False



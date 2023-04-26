import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import argparse
import datetime
import json
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np

from reimplemented_libraries import CxnUtils
from py4j.java_gateway import JavaGateway, GatewayParameters
from rdkit import Chem, RDLogger

from mcts_main import Mcts
from mcts_modules import print_route, save_route
from model_modules import load_model, predict_templates
from utils import *
from preprocessing_modules import atom_features, build_data


def get_parser():
    """ Parse arguments
    Args:
    Returns:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='description',
        usage='usage'
    )
    parser.add_argument(
        "-a", "--max_atom_num", required=False, type=int,
        help="Max number of atoms in a molecule"
    )
    parser.add_argument(
        "-c", "--search_count", required=False, type=int,
        help="MCTS max search count"
    )
    parser.add_argument(
        "--config", type=str,
        help="path to config file"
    )
    parser.add_argument(
        "-d", "--rollout_depth", required=False, type=int,
        help="Rollout max depth count"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode"
    )
    parser.add_argument(
        "-e", "--expansion_model", required=False, type=str,
        help="Path to expansion model file"
    )
    parser.add_argument(
        "-er", "--expansion_rules", required=False, type=str,
        help="Path to reaction rules for expansion"
    )
    parser.add_argument(
        "-f", "--descriptor", required=False, type=str,
        help="Specify ECFP or GCN"
    )
    parser.add_argument(
        '--gcn_expansion_config', required=False, type=str,
        help='Path to GCN expansion config file'
    )
    parser.add_argument(
        '--gcn_rollout_config', required=False, type=str,
        help='Path to GCN rollout config file'
    )
    parser.add_argument(
        "--template_scores", required=False, type=str,
        help="Path to template scores file"
    )
    parser.add_argument(
        "-m", "--starting_material", required=False, type=str,
        help="Path to starting materials file"
    )
    parser.add_argument(
        "-i", "--intermediate_material", required=False, type=str,
        help="Path to intermediate materials file"
    )
    parser.add_argument(
        "-p", "--rollout_model", required=False, type=str,
        help="Path to rollout model file"
    )
    parser.add_argument(
        "-pr", "--rollout_rules", required=False, type=str,
        help="Path to reaction rules for playout"
    )
    parser.add_argument(
        "-is", "--in_scope_model", required=False, type=str,
        help="Path to In scope model weights"
    )
    parser.add_argument(
        "-r", "--save_result_dir", type=str, default="result",
        help="Path to a result directory"
    )
    parser.add_argument(
        "-t", "--target", required=False, type=str,
        help="Path to target molecule file"
    )
    parser.add_argument(
        "-k", "--knowledge", required=False, nargs="+", default=[], type=str,
        choices=["cdscore", "rdscore", "asscore", "stscore", "intermediate_score", "template_score", "all"],
        help="choice chemical knowledges"
    )
    parser.add_argument(
        "--knowledge_weights", required=False, nargs=6, default=[1., 1., 1., 1., 1., 1.], type=float,
        help="knowledge score's weights in selection. [cdscore, rdscore, asscore, stscore, intermediate_score, template_score]"
    )
    parser.add_argument(
        "--save_tree", required=False, action='store_true', default=False,
        help="save searched tree information"
    )
    parser.add_argument(
        "--sel_const", required=False, default=3, type=int,
        help="constant value for selection"
    )
    parser.add_argument(
        "--expansion_num", required=False, type=int, default=50,
        help="the number of expanded nodes during the expansion step"
    )
    parser.add_argument(
        "--time_limit", type=int, default=0,
        help="Time limit for search over one molecule"
    )
    parser.add_argument(
        "--cum_prob_mod", action="store_true", default=False,
        help="uses cumulative probabilities to prune reaction candidates"
    )
    parser.add_argument(
        "--cum_prob_thresh", required=False, type=float, default=0.955,
        help="threshold used for cumulative probabilities mod"
    )
    parser.add_argument(
        "--chem_axon", action="store_true",
        help="Uses chem axon dependencies"
    )
    parser.add_argument(
        "--random_seed", required=False, type=int, default=0,
        help="Fix random seed (has to be different than 0 to be activated)"
    )
    return parser.parse_args()


def get_logger(level, save_dir):
    # logger
    logger = getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False
    # formatter
    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s ")
    # handler
    fh = FileHandler(filename=os.path.join(save_dir, "run.log"), mode='w')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    sh = StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    #
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    config = get_config(args)
    if args.random_seed:
        random.seed(args.random_seed)
    # Create save directory
    now = datetime.datetime.now()
    name_stem = config["target"].split('/')[-1].split('.')[0]
    config["save_result_dir"] = os.path.join(config["save_result_dir"], f"{name_stem}_{now:%Y%m%d%H%M}")
    os.makedirs(config["save_result_dir"], exist_ok=True)

    # Save parameters
    with open(os.path.join(config["save_result_dir"], "parameters.json"), 'w') as f:
        json.dump({k: repr(v) for k, v, in config.items()}, f, indent=2)

    # Setup logger
    level = DEBUG if args.debug else INFO
    logger = get_logger(level, config["save_result_dir"])
    if not args.debug:
        RDLogger.DisableLog("rdApp.*")

    if args.chem_axon:
        # Setup JVM
        gateway_port = 25333 + np.random.randint(1, 3000)
        while is_port_in_used(gateway_port):
            gateway_port += 1
        proxy_port = gateway_port + 1
        while is_port_in_used(proxy_port):
            proxy_port += 1
        logger.info(f"gateway port: {gateway_port} proxy port: {proxy_port}\n")

        cmd = f"java CxnUtils {gateway_port} {config['rollout_rules']}"
        subprocess.Popen(cmd.split(" "))
        time.sleep(3)
        gateway = JavaGateway(start_callback_server=True,
                              python_proxy_port=proxy_port,
                              gateway_parameters=GatewayParameters(port=gateway_port, auto_convert=True))
        logger.info("Start up java gateway")
    else:
        gateway = CxnUtils(config['rollout_rules'])

    try:
        # data preparation
        target_mol = Chem.MolFromMolFile(config['target'])
        if target_mol is None:
            logger.error("Can't read the input molecule file. Please check it.")
            sys.exit(1)
        expansion_rules = ReactionUtils.get_reactions(config['expansion_rules'], config['save_result_dir'])
        rollout_rules = ReactionUtils.get_reactions(config['rollout_rules'], config['save_result_dir'])
        with open(config['starting_material'], 'r') as f:
            start_materials = set([s.strip() for s in f.readlines()])
            if all([len(x) == 27 for x in start_materials]):
                start_materials.add("inchi") # Sets start materials format to inchi
        if config['intermediate_material'] is not None:
            with open(config['intermediate_material'], 'r') as f:
                intermediate_materials = set([s.strip() for s in f.readlines()])
        else:
            intermediate_materials = set()
        if config['descriptor'] == 'ECFP':
            expansion_model = load_model('expansion', config, class_num=len(expansion_rules))
            rollout_model = load_model('rollout', config, class_num=len(rollout_rules))
        elif config['descriptor'] == 'GCN':
            expansion_model = load_model('expansion', config, class_num=len(expansion_rules))
            rollout_model = load_model('rollout', config, class_num=len(rollout_rules))
        else:
            logger.error("set 'descriptor' to GCN or ECFP")
            sys.exit(1)
        if config["template_scores"]:
            template_scores = json.load(open(config["template_scores"], "r"))
        else:
            template_scores = {}
        in_scope_model = load_model('in_scope', config)
        # main process
        mcts = Mcts(target_mol, expansion_rules, rollout_rules, start_materials, intermediate_materials, template_scores, config)

        logger.info(f"[INFO] knowledge type: {config['knowledge']}")
        logger.info("[INFO] start search")
        start = time.time()
        leaf_node, is_proven = mcts.search(expansion_model, rollout_model, in_scope_model, logger, gateway=gateway, time_limit=args.time_limit)
        elapsed_time = time.time() - start
        logger.info(f"[INFO] done in {elapsed_time:5f} s")

        with open(os.path.join(config['save_result_dir'], "time.txt"), 'w') as f:
            f.write(f"{elapsed_time}")

        nodes = []
        if leaf_node is None:
            Path(os.path.join(config['save_result_dir'], "not_proven")).touch()
            logger.info("Can't apply any predicted reaction templates to the target compound.")
            sys.exit()
        while leaf_node.parent_node is not None:
            nodes.append(leaf_node)
            leaf_node = leaf_node.parent_node
        else:
            nodes.append(leaf_node)
        print_route(nodes, is_proven, logger)
        save_route(nodes, config['save_result_dir'], is_proven, config["knowledge_weights"])
    finally:
        if args.chem_axon:
            gateway.shutdown()
            logger.info("Shutdown java gateway")


if __name__ == "__main__":
    main()

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

from py4j.java_gateway import JavaGateway, GatewayParameters
from rdkit import Chem, RDLogger

from mcts_main import Mcts
from mcts_modules import print_route, save_route
from model_modules import load_model
from utils import is_port_in_used, ReactionUtils, get_default_config


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
        "-c", "--search_count", required=False, type=int, default=100,
        help="the maximum number of iterations of MCTS"
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
        "-m", "--starting_material", required=False, type=str,
        help="Path to starting materials file"
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
        "-r", "--save_result_dir", type=str, default="result",
        help="Path to a result directory"
    )
    parser.add_argument(
        "-t", "--target", required=False, type=str,
        help="Path to target molecule file"
    )
    parser.add_argument(
        "-k", "--knowledge", required=False, nargs="+", default=[], type=str,
        choices=["cdscore", "rdscore", "asscore", "stscore", "all"],
        help="choice chemical knowledges"
    )
    parser.add_argument(
        "--knowledge_weights", required=False, nargs=4, default=[1., 1., 1., 1.], type=float,
        help="knowledge score's weights in selection. [cdscore, rdscore, asscore, stscore]"
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

    # Setup config: Arguments take priority over config file
    config = get_default_config()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    config['max_atom_num'] = int(args.max_atom_num or config['max_atom_num'])
    config['search_count'] = int(args.search_count or config['search_count'])
    config['rollout_depth'] = int(args.rollout_depth or config['rollout_depth'])
    config['expansion_model'] = args.expansion_model or config['expansion_model']
    config['expansion_rules'] = args.expansion_rules or config['expansion_rules']
    config['rollout_model'] = args.rollout_model or config['rollout_model']
    config['rollout_rules'] = args.rollout_rules or config['rollout_rules']
    config['descriptor'] = args.descriptor or config['descriptor']
    config['gcn_expansion_config'] = args.gcn_expansion_config or config['gcn_expansion_config']
    config['gcn_rollout_config'] = args.gcn_rollout_config or config['gcn_rollout_config']
    config['starting_material'] = args.starting_material or config['starting_material']
    config['save_result_dir'] = args.save_result_dir or config['save_result_dir']
    config['target'] = args.target or config['target']
    config['knowledge'] = set(args.knowledge)
    config["knowledge_weights"] = args.knowledge_weights
    config['save_tree'] = args.save_tree
    config['selection_constant'] = args.sel_const
    config['expansion_num'] = args.expansion_num

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

    # Setup JVM
    gateway_port = 25333 + random.randint(1, 3000)
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
        if config['descriptor'] == 'ECFP':
            expansion_model = load_model('expansion', config, class_num=len(expansion_rules))
            rollout_model = load_model('rollout', config, class_num=len(rollout_rules))
        elif config['descriptor'] == 'GCN':
            expansion_model = load_model('expansion', config, class_num=len(expansion_rules))
            rollout_model = load_model('rollout', config, class_num=len(rollout_rules))
        else:
            logger.error("set 'descriptor' to GCN or ECFP")
            sys.exit(1)
        # main process
        mcts = Mcts(target_mol, expansion_rules, rollout_rules, start_materials, config)

        logger.info(f"[INFO] knowledge type: {config['knowledge']}")
        logger.info("[INFO] start search")
        start = time.time()
        leaf_node, is_proven = mcts.search(expansion_model, rollout_model, logger, gateway=gateway)
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
        gateway.shutdown()
        logger.info("Shutdown java gateway")


if __name__ == "__main__":
    main()

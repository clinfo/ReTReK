import json
import sys

from keras import models
from rdkit import Chem
import tensorflow as tf

from kgcn.core import CoreModel
from kgcn.data_util import build_data
from kgcn.gcn import get_default_config, load_model_py


def load_model(model_name, config, class_num=None):
    """ Load trained model of Tensorflow or Keras
    Args:
        model_name (str):
        config (dict):
        class_num (int):
    Returns:
        Loaded model instance
    """
    model = None
    if config['descriptor'] == 'ECFP':
        if model_name == 'expansion':
            model = models.load_model(config['expansion_model'])
        elif model_name == 'rollout':
            model = models.load_model(config['rollout_model'])

    if config['descriptor'] == 'GCN':
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with graph.as_default():
            from utils import MoleculeUtils
            # dummy setting
            mol = Chem.MolFromSmiles('C1=CC=CC=C1')
            input_data = None
            trained_model_path = None
            if model_name == 'expansion':
                gcn_config = get_config(config['gcn_expansion_config'])
                input_data = MoleculeUtils.generate_gcn_descriptor(mol, config['max_atom_num'], class_num)
                trained_model_path = config['expansion_model']
            elif model_name == 'rollout':
                gcn_config = get_config(config['gcn_rollout_config'])
                input_data = MoleculeUtils.generate_gcn_descriptor(mol, config['max_atom_num'], class_num)
                trained_model_path = config['rollout_model']
            _, info = build_data(gcn_config, input_data, verbose=False)
            model = CoreModel(sess, gcn_config, info)
            load_model_py(model, gcn_config["model.py"], is_train=False)
            # Initialize session
            saver = tf.train.Saver()
            saver.restore(sess, trained_model_path)
    return model


def predict_templates(model, input_data, model_name, config):
    """ Predict which reactions probably occured using trained model
    Args:
        model:
        input_data: numpy array for ECFP, dict[adj, label, feature, mask_label, max_node_num] for GCN
        model_name (str):
        config (dict):
    Returns:
        Numpy array type predicted values
    """
    if config['descriptor'] == 'ECFP':
        preds = model.predict_proba(input_data)
        return preds[0]
    if config['descriptor'] == 'GCN':
        if model_name == 'expansion':
            config = get_config(config['gcn_expansion_config'])
        elif model_name == 'rollout':
            config = get_config(config['gcn_rollout_config'])
        else:
            print("[ERROR] check model_name")
            sys.exit(1)
        all_data, info = build_data(config, input_data, verbose=False)
        _, _, prediction_data = model.pred_and_eval(all_data, local_init=False)
        return prediction_data[0]


def get_config(path):
    """ Get GCN configuration
    Args:
        path (str): Path to GCN config file
    """
    config = get_default_config()
    with open(path, "r") as fp:
        config.update(json.load(fp))
    return config


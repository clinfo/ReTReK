import json
import sys

from keras import models
from rdkit import Chem
import tensorflow as tf
import numpy as np
import torch

from kgcn.core import CoreModel
from kgcn.gcn import get_default_config, load_model_py

from preprocessing_modules import build_data
from model.in_scope_filter import InScopeModel

class CoreModelInference(CoreModel):
    """
    Subclass of CoreModel from kGCN to allow inference only mode.
    """
    def build(self,model,is_train=True,feed_embedded_layer=False,batch_size=None):
        #
        config=self.config
        info=self.info
        if batch_size is None:
            batch_size=config["batch_size"]
        #
        info.param=None
        if config["param"] is not None:
            if type(config["param"]) is str:
                print("[LOAD] ",config["param"])
                fp = open(config["param"], 'r')
                info.param=json.load(fp)
            else:
                info.param=config["param"]

        # feed_embedded_layer=True => True emmbedingレイヤを使っているモデルの可視化。IGはemmbedingレイヤの出力を対象にして計算される。
        self.placeholders = model.build_placeholders(info, config, batch_size=batch_size, feed_embedded_layer=feed_embedded_layer, inference_only=True)
        _model,self.prediction = model.build_model(self.placeholders,info,config,batch_size=batch_size, feed_embedded_layer=feed_embedded_layer, inference_only=True)
        self.nn=_model
        if _model is not None and hasattr(_model,'out'):
            self.out=_model.out
        else:
            # Deprecated: for old version
            self.out=_model

    def pred(self,data, local_init=True):
        sess=self.sess
        config=self.config
        info=self.info
        batch_size=config["batch_size"]
        # start
        data_idx=list(range(data.num))
        itr_num=int(np.ceil(data.num/batch_size))
        prediction_data=[]

        if local_init:
            local_init_op = tf.local_variables_initializer()
            sess.run(local_init_op)
        for itr in range(itr_num):
            offset_b=itr*batch_size
            batch_idx=data_idx[offset_b:offset_b+batch_size]
            feed_dict=self.construct_feed(batch_idx,self.placeholders,data,batch_size=batch_size,is_train=False,info=info,config=config)
            out_prediction=sess.run(self.prediction, feed_dict=feed_dict)
            prediction_data.extend(out_prediction[:len(batch_idx)])

        return prediction_data

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
    if model_name == "in_scope":
        if config["in_scope_model"]:
            model = InScopeModel()
            model.load_state_dict(torch.load(config["in_scope_model"], map_location=torch.device('cpu')))
            model.eval()
            return model
        else:
            return None
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
            _, info = build_data(gcn_config, input_data, return_info=True)
            model = CoreModelInference(sess, gcn_config, info)
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
        all_data = build_data(config, input_data, return_info=False)
        prediction_data = model.pred(all_data, local_init=False)
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


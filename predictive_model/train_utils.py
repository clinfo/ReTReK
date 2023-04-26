import tensorflow as tf
import os
import datetime
import numpy as np
import json
from kgcn.gcn import get_default_config


class EarlyStopping:
    """
    Handles early stopping and saving checkpoints with best score.
    """

    def __init__(self, config, monitor="validation_cost", comp=np.less):
        self.best_validation_cost = None
        self.validation_count = 0
        self.config = config
        self.monitor = monitor
        self.comp = comp

    def evaluate_validation(self, saver, sess, config, k_fold_num, validation_result):
        config = self.config
        if self.best_validation_cost is not None and self.comp(
            self.best_validation_cost, validation_result[self.monitor]
        ):
            self.validation_count += 1
            if config["patience"] > 0 and self.validation_count >= config["patience"]:
                self.print_info(validation_result)
                print("[Early stop] by validation")
                return True
        else:
            self.validation_count = 0
            self.best_validation_cost = validation_result[self.monitor]
            save_restore_ckpt(
                saver, sess, "save", config["save_path"], k_fold_num, "best"
            )
        self.print_info(validation_result)
        return False

    def print_info(self, info):
        config = self.config
        epoch = info["epoch"]
        training_cost = info["training_cost"]
        training_accuracy = info["training_accuracy"]
        training_auc = info["training_auc"]
        training_precision = info["training_precision"]
        training_recall = info["training_recall"]
        training_fscore = info["training_fscore"]
        training_acc_step = info["training_accuracy_step"]

        val_cost = info["validation_cost"]
        val_accuracy = info["validation_accuracy"]
        val_auc = info["validation_auc"]
        val_precision = info["validation_precision"]
        val_recall = info["validation_recall"]
        val_fscore = info["validation_fscore"]
        val_acc_step = info["validation_accuracy_step"]
        format_tuple = (
            training_cost,
            training_accuracy,
            training_auc,
            training_precision,
            training_recall,
            training_fscore,
            training_acc_step,
            val_cost,
            val_accuracy,
            val_auc,
            val_precision,
            val_recall,
            val_fscore,
            val_acc_step,
            self.validation_count,
        )
        print(
            "TRAINING: loss %g, Acc %g, AUC %g, Precision %g, Recall %g, FScore %g, Acc step %g \n"
            "VALIDATION loss %g, Acc %g, AUC %g, Precision %g, Recall %g, FScore %g, Acc step %g  (count=%d) "
            % format_tuple
        )


class ExpAvgMeter:
    """
    Class used for computing running exponential mean.
    """

    def __init__(self, coeff):
        """
        Args:
            coeff (float): coefficient used for exponential moving average.
        """
        self.coeff = coeff
        self.value = 0
        self.iter = 0

    def update(self, value):
        self.iter += 1
        if self.iter == 1:
            self.value = value
        else:
            self.value = self.coeff * self.value + (1 - self.coeff) * value


def get_config(path):
    """Get GCN configuration
    Args:
        path (str): Path to GCN config file
    """
    config = get_default_config()
    with open(path, "r") as fp:
        config.update(json.load(fp))
    return config


def init_var(def_value=0, dtype=tf.float32):
    return tf.Variable(def_value, dtype=dtype)


def init_summary(var_names, log_tb, sess, create_valid):
    """
    Creates tensorflow summary for logging to tensorboard
    Args:
        var_names (list): list of str corresponding to variables to track
        log_tb (str): path to save tensorboard logs to
        sess (tf.Session): session to use
        create_valid (bool): create validation summary writer or not
    """
    tracked = {}
    for var in var_names:
        if isinstance(var, tuple):
            var, args = var
            tracked[var] = init_var(*args)
        else:
            tracked[var] = init_var()
        tf.summary.scalar(var, tracked[var])
    merged = tf.summary.merge_all()
    os.makedirs(log_tb, exist_ok=True)
    log_dir = os.path.join(log_tb, f"{datetime.datetime.now():%Y%m%d%H%M}")
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, "train"), sess.graph)
    val_writer = None
    if create_valid:
        val_writer = tf.summary.FileWriter(os.path.join(log_dir, "val"), sess.graph)
    print(f"Logging to : {log_dir}")
    return merged, tracked, train_writer, val_writer


def save_restore_ckpt(saver, sess, mode, save_path, k_fold_num=None, suffix=""):
    """
    Saves or restores checkpoint
    Args:
        saver (tf.train.Saver): used to save and restore checkpoint
        sess (tf.Session): session to use
        mode (str): 'save' or 'restore'
        save_path (str): path to directory to save model to or load model from
        k_fold_num (int): k fold identifiyer
        suffix (str): suffix to identify checkpoint
    """
    if k_fold_num is not None:
        path = os.path.join(save_path, f"model.{k_fold_num:03d}.{suffix}.ckpt")
    else:
        path = os.path.join(save_path, f"model.{suffix}.ckpt")
    print(f"[{mode.upper()}] ", path)
    if mode == "save":
        saver.save(sess, path)
    elif mode == "restore":
        saver.restore(sess, path)


def aggregate_metrics(metrics, data, key_prefix):
    """
    Aggregate metrics
    Returns updated metrics.
    Args:
        metrics (dict): metrics to evaluate
        data (dotdict): data used in training
        key_prefix (str): 'training_' or 'validation_'
    """
    reduce_sum = ["correct_count", "correct_count_steps"]
    reduce_mean = ["mse", "mae", "r2", "auc", "precision", "recall", "fscore"]
    aggregated_metrics = {}
    for m in reduce_sum:
        aggregated_metrics[key_prefix + m] = np.sum([met[m] for met in metrics], axis=0)
    for m in reduce_mean:
        aggregated_metrics[key_prefix + m] = np.mean(
            [met[m] for met in metrics], axis=0
        )
    aggregated_metrics[key_prefix + "accuracy"] = (
        aggregated_metrics[key_prefix + "correct_count"] / data.num
    )
    aggregated_metrics[key_prefix + "accuracy_step"] = (
        aggregated_metrics[key_prefix + "correct_count_steps"] / data.num
    )
    return aggregated_metrics


def update_metrics(model, cost, metrics, data, key_prefix):
    """
    Evaluate metrics and update metrics stored in model.
    Returns updated metrics.
    Args:
        model (kgcn DefaultModel): model used in training
        cost (float): cost value to add to model cost_list
        metrics (dict): metrics to evaluate
        data (dotdict): data used in training
        key_prefix (str): 'training_' or 'validation_'
    """
    aggregated_metrics = aggregate_metrics(metrics, data, key_prefix)

    cost_list = getattr(model, key_prefix + "cost_list")
    metrics_list = getattr(model, key_prefix + "metrics_list")
    cost_list.append(cost)
    metrics_list.append(aggregated_metrics)
    return aggregated_metrics


def build_optimizer(
    cost,
    learning_rate,
    schedule,
    graph,
    epoch,
    step_per_epoch,
    step_div,
    div_factor=10,
    warmup=True,
    warmup_div_fact=100,
    warmup_step=100,
):
    """
    Handles optimizer and learing rate schedule.
    Args:
        cost (tf.Tensor): tensor containing value to minimize
        learning_rate (float): base learning rate value
        schedule (str): learning rate schedule type
        graph (tf.Graph): graph to which attach global step
        epoch (int): number of epochs of the training run
        step_per_epoch (int): number of batches in one epoch
        step_div (int): number of steps in between each step_scheduler div
        div_factor (float): factor to apply at each step_scheduler division
        warmup (bool): use warmup of learning rate
        warmup_div_fact (int): value to determine starting learning rate
        warmup_step (int): number of steps of warmup procedure
    """
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.train.get_global_step(graph)
    with tf.control_dependencies(update_ops):
        if warmup:
            warmup_lr = (
                learning_rate
                / warmup_div_fact
                * np.exp(np.log(warmup_div_fact) / warmup_step)
                ** tf.cast(global_step, tf.float32)
            )
            learning_rate = tf.minimum(warmup_lr, learning_rate)
            step = tf.maximum(
                global_step - warmup_step, 0
            )  # delays scheduler update till after warmup_step
        if schedule == "cosine":
            learning_rate = tf.train.cosine_decay(
                learning_rate, step, epoch * step_per_epoch
            )
        elif schedule == "step":
            epoch_nb = step // step_per_epoch
            scaler = tf.cast(div_factor ** (epoch_nb // step_div), tf.float32)
            learning_rate = learning_rate / scaler
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(
            cost, global_step=global_step
        )
    return train_step, learning_rate

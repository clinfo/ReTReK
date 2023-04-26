import argparse
import tensorflow as tf
import json
import os
import time
import numpy as np
import datetime
import optuna
import pandas as pd
from tqdm import tqdm
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from kgcn.core import CoreModel
from kgcn.gcn import load_model_py

from data import build_data, construct_feed
from train_utils import *
from hyper_param_opt import get_objective, print_study_results, metrics_to_score


def get_parser():
    """Parse arguments
    Args:
    Returns:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="description", usage="usage")
    parser.add_argument("job")
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Path to config file"
    )
    parser.add_argument("-d", "--data", required=True, type=str, help="Path to dataset")
    parser.add_argument(
        "--tensorboard",
        required=False,
        type=str,
        default="",
        help="Tensorboard logging dir, if nothing passed then not used",
    )
    parser.add_argument(
        "--cat_dim",
        required=False,
        type=int,
        default=20,
        help="For evaluation, total number of steps categories (should be based on training set)",
    )
    parser.add_argument(
        "--max_time",
        required=False,
        type=float,
        default=None,
        help="For evaluation, maximum processing time (should be based on training set)",
    )

    return parser.parse_args()


def fit_epoch(
    model,
    epoch,
    data,
    batch_size,
    data_idx,
    sess,
    graph,
    config,
    mode="train",
    log_tb=False,
    meters={},
    tracked=None,
    merged=None,
    train_writer=None,
):
    """
    Handles processing of one epoch in training or validation mode.
    Args:
        model (kgcn DefaultModel): model to use to process data
        epoch (int): epoch number
        data (dotdict): dictionnary containing data
        batch_size (int): size of batches
        data_idx (list): list of indexes of samples to fetch
        sess (tf.Session): session to use
        graph (tf.Graph): graph to use
        config (dict): dictionnary containing meta information on training
        mode (str): 'train' or 'validation'
        log_tb (str): path to save tensorboard logs to, if not passed then no logging
        meters (dict): dictionnary containing output metrics as keys and average meters as values
        tracked (dict): dictionnary used to store variables tracked by tensorboard
        merged (tf.summary): Used to log to tensoboard
        train_writer (tf.summary.FileWriter): Used to log to tensorboard
    Returns:
        cost (float): Average cost for the epoch
        metrics (list): List of computed metrics
        step (int): Number of processed batches since beginning of training
        predictions (np.ndarray): model predictions
        labels (np.ndarray): data labels
        smiles (np.ndarray): smiles of samples
    """
    itr_num = int(np.ceil(data.num / batch_size))
    cost = 0
    metrics = []
    predictions = []
    labels = []
    smiles = []
    pbar = tqdm(range(itr_num))
    dropout, is_train = (config["dropout"], True) if mode == "train" else (0.0, False)
    step = None
    for itr in pbar:
        offset_b = itr * batch_size
        batch_idx = data_idx[offset_b : offset_b + batch_size]
        feed_dict = construct_feed(
            batch_idx,
            model.placeholders,
            data,
            batch_size,
            dropout,
            is_train,
            info=model.info,
        )
        if mode == "train":
            # running parameter update with tensorflow
            _, out_cost_sum, out_metrics, prediction = sess.run(
                [model.train_step, model.cost_sum, model.metrics, model.prediction],
                feed_dict=feed_dict,
                options=None,
                run_metadata=None,
            )
        else:
            out_cost_sum, out_metrics, prediction = sess.run(
                [model.cost_sum, model.metrics, model.prediction], feed_dict=feed_dict
            )
        predictions.append(prediction)
        labels.append(data.labels[batch_idx])
        smiles.append(data.smiles[batch_idx])
        cost += out_cost_sum
        metrics.append(out_metrics)
        if mode == "train":
            step = sess.run(tf.train.get_global_step(graph))
            for metric_key, meter in meters.items():
                meter.update(out_metrics[metric_key])
            pbar.set_description(
                "Train Epoch : {0}/{1} Loss : {2:.4f} | Acc : {3:.2f} | Acc Step : {4:.2f}".format(
                    epoch + 1,
                    config["epoch"],
                    meters["loss"].value,
                    meters["accuracy"].value,
                    meters["accuracy_step"].value,
                )
            )
            if log_tb and step % 50 == 0:
                sess.run(
                    [
                        tracked["Accuracy"].assign(meters["accuracy"].value),
                        tracked["Loss"].assign(meters["loss"].value),
                        tracked["LR"].assign(model.learning_rate),
                    ]
                )
                train_writer.add_summary(sess.run(merged), step)

    cost /= data.num
    labels = np.concatenate(labels)
    smiles = np.concatenate(smiles)
    predictions = np.concatenate(predictions)[: len(labels)]
    auc = roc_auc_score(labels[:, 0], predictions[:, 0])
    for m in metrics:
        m["auc"] = auc
        precision, recall, fscore, _ = precision_recall_fscore_support(
            labels[:, 0],
            np.where(predictions[:, 0] > 0.5, 1, 0),
            average="binary",
            pos_label=1,
        )
        m["precision"] = precision
        m["recall"] = recall
        m["fscore"] = fscore

    return cost, metrics, step, predictions, labels, smiles


def fit(model, graph, train_data, valid_data, log_tb="", k_fold_num=None):
    """
    Handles training of a model
    Args
        model (kgcn DefaultModel): model to train
        graph (tf.Graph): graph to use for training
        train_data (dotdict): training data
        valid_data (dotdict): validation data
        log_tb (str): path to save tensoboard logs to. No logging if not passed.
        k_fold_num (int): K fold identifiyer, None if cross validation not used
    Returns:
        validation_result_list (list): list of validation metrics
    """
    # Init variables
    sess = model.sess
    config = model.config
    batch_size = config["batch_size"]
    model.training_cost_list, model.training_metrics_list = [], []
    model.validation_cost_list, model.validation_metrics_list = [], []
    train_writer = None
    tracked = None
    merged = None
    if log_tb:
        keep_track = ["Loss", "Accuracy", "AUC", ("LR", (config["learning_rate"],))]
        merged, tracked, train_writer, val_writer = init_summary(
            keep_track, log_tb, sess, valid_data is not None
        )

    saver = tf.train.Saver(max_to_keep=None)
    if config["retrain"] is None:
        sess.run(tf.global_variables_initializer())
    else:
        print("[LOAD]", config["retrain"])
        saver.restore(sess, config["retrain"])

    print("# Train data samples = ", train_data.num)
    if valid_data is not None:
        print("# Valid data samples = ", valid_data.num)

    early_stopping = EarlyStopping(config, monitor="validation_cost", comp=np.less)

    train_idx = list(range(train_data.num))
    if valid_data is not None:
        valid_idx = list(range(valid_data.num))

    best_score = None
    best_result = None
    validation_result_list = []
    os.makedirs(config["save_path"], exist_ok=True)
    meters = {
        "loss": ExpAvgMeter(0.95),
        "accuracy": ExpAvgMeter(0.95),
        "accuracy_step": ExpAvgMeter(0.95),
    }

    # Train model
    for epoch in range(config["epoch"]):
        # Init epoch
        np.random.shuffle(train_idx)
        local_init_op = tf.local_variables_initializer()
        # training
        sess.run(local_init_op)
        training_cost, training_metrics, step, _, _, _ = fit_epoch(
            model,
            epoch,
            train_data,
            batch_size,
            train_idx,
            sess,
            graph,
            config,
            "train",
            log_tb,
            meters,
            tracked,
            merged,
            train_writer,
        )
        # validation
        sess.run(local_init_op)
        validation_cost, validation_metrics, _, _, _, _ = fit_epoch(
            model,
            epoch,
            valid_data,
            batch_size,
            valid_idx,
            sess,
            graph,
            config,
            "validation",
        )

        # evaluation and recording costs and accuracies
        training_metrics = update_metrics(
            model, training_cost, training_metrics, train_data, "training_"
        )
        validation_metrics = update_metrics(
            model, validation_cost, validation_metrics, valid_data, "validation_"
        )

        # check point
        save_path = None
        if (epoch) % config["save_interval"] == 0:
            # save
            save_restore_ckpt(
                saver,
                sess,
                "save",
                config["save_path"],
                k_fold_num,
                f"{epoch:05d}",
            )
        # early stopping and printing information
        validation_result = {
            "epoch": epoch + 1,
            "validation_cost": validation_cost,
            "training_cost": training_cost,
            "save_path": save_path,
        }
        validation_result.update(validation_metrics)
        if training_metrics is not None:
            validation_result.update(training_metrics)
        validation_result_list.append(validation_result)

        # Early stopping
        if early_stopping.evaluate_validation(
            saver, sess, config, k_fold_num, validation_result
        ):
            break
        if np.isnan(validation_cost):
            break
        if log_tb and valid_data is not None:
            sess.run(
                [
                    tracked["Accuracy"].assign(
                        validation_result["validation_accuracy"]
                    ),
                    tracked["Loss"].assign(validation_result["validation_cost"]),
                    tracked["AUC"].assign(validation_result["validation_auc"]),
                ]
            )
            val_writer.add_summary(sess.run(merged), step)
            sess.run(
                [tracked["AUC"].assign(training_metrics["training_auc"])]
            )  # restores training AUC value for tensorboard plots

    # saving last model
    save_restore_ckpt(
        saver, sess, "save", config["save_path"], k_fold_num, "last"
    )

    # restore best
    save_restore_ckpt(
        saver, sess, "restore", config["save_path"], k_fold_num, "best"
    )

    if log_tb:
        train_writer.flush()
        train_writer.close()
        if valid_data is not None:
            val_writer.flush()
            val_writer.close()

    return validation_result_list


def train(args, sess, graph, config, return_score=False):
    """
    Train a model according to the parameters defined in args and config.
    Args
        args (argparse.Namespace): command line arguments
        sess (tf.Session): session to use
        graph (tf.Graph): graph to use
        config (dict): contains parameters for training
        return_score (bool): if set to True, returns average of all metrics
    Returns:
        if return_score returns score (float)
    """

    # Loading of data
    dataset = json.load(open(args.data, "r"))
    indices = np.arange(len(dataset["features"]))
    train_index, valid_index = train_test_split(
        indices, train_size=0.8, shuffle=True, random_state=1234
    )
    train_set, valid_set = {}, {}
    for k in dataset.keys():
        train_set[k] = [dataset[k][i] for i in train_index]
        valid_set[k] = [dataset[k][i] for i in valid_index]
    train_data, info = build_data(train_set, True, config["normalize_adj_flag"])
    valid_data, valid_info = build_data(valid_set, True, config["normalize_adj_flag"])
    info.cat_dim = max(info.cat_dim, valid_info.cat_dim)

    # Building model
    model = CoreModel(sess, config, info)
    load_model_py(model, config["model.py"])

    # Building optimizer
    step_per_epoch = int(np.ceil(train_data.num / config["batch_size"]))
    tf.train.create_global_step(graph)
    model.train_step, model.learning_rate = build_optimizer(
        cost=model.cost,
        learning_rate=config["learning_rate"],
        schedule=config["lr_schedule"],
        graph=graph,
        epoch=config["epoch"],
        step_per_epoch=step_per_epoch,
        step_div=config["step_div"],
        div_factor=config["div_factor"],
    )

    # Training
    start_t = time.time()
    fit(
        model=model,
        graph=graph,
        train_data=train_data,
        valid_data=valid_data,
        log_tb=args.tensorboard,
    )
    train_time = time.time() - start_t
    print(f"training time: {train_time}[sec]")

    # Validation
    start_t = time.time()
    valid_cost, valid_metrics, _, _, _, _ = fit_epoch(
        model,
        0,
        valid_data,
        config["batch_size"],
        list(range(valid_data.num)),
        sess,
        graph,
        config,
        "validation",
    )
    valid_metrics = aggregate_metrics(valid_metrics, valid_data, "")
    infer_time = time.time() - start_t
    print(f"final cost = {valid_cost}\n")
    for metric_name in valid_metrics:
        print(f"{metric_name} = {valid_metrics[metric_name]}")
    print(f"validation time: {infer_time}[sec]\n")

    if return_score:
        return metrics_to_score(valid_metrics, config)


def evaluate(args, sess, graph, config):
    """
    Evaluate a model according to the parameters defined in args and config.
    Args
        args (argparse.Namespace): command line arguments
        sess (tf.Session): session to use
        graph (tf.Graph): graph to use
        config (dict): contains parameters for training
    Returns:
        Saves results to json file and predictions to csv file in directory specified by save_path in config.
    """

    dataset = json.load(open(args.data, "r"))
    data, info = build_data(
        dataset,
        True,
        config["normalize_adj_flag"],
        cat_dim=args.cat_dim,
        max_time=args.max_time,
    )

    model = CoreModel(sess, config, info)
    load_model_py(model, config["model.py"])
    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("No checkpoint_path found in config. Cannot load weights.")
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    cost, metrics, _, predictions, labels, smiles = fit_epoch(
        model,
        0,
        data,
        config["batch_size"],
        list(range(data.num)),
        sess,
        graph,
        config,
        "validation",
    )
    metrics = aggregate_metrics(metrics, data, "")
    print(f"final cost = {cost}\n")
    for metric_name in metrics:
        print(f"{metric_name} = {metrics[metric_name]}")

    for metric_name, metric in metrics.items():
        if isinstance(metric, np.ndarray):
            metrics[metric_name] = metric.astype(float).tolist()
        else:
            metrics[metric_name] = float(metric)
    with open(os.path.join(config["save_path"], "results.json"), "w") as f:
        json.dump(metrics, f)

    df_dict = {}
    for k, v in info.outputs_mapper.items():
        if type(v) == list:
            df_dict[k] = np.argmax(predictions[:, v[0] : v[1]], axis=1).tolist()
        else:
            df_dict[k] = predictions[:, v].tolist()
    for k, v in info.labels_mapper.items():
        df_dict[f"GT_{k}"] = labels[:, v].tolist()
    smiles = smiles.reshape(-1)
    df_dict["smiles"] = smiles.tolist()
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(os.path.join(config["save_path"], "evaluation_data.csv"))


def main():
    args = get_parser()
    config = get_config(args.config)

    if args.job == "train":
        if config["hyper_param_opt"]:
            tune_params = config["tune_params"]
            objective = get_objective(train, tune_params, args, config)
            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective,
                n_trials=config["n_trials"],
                callbacks=[
                    lambda study, trial: study.trials_dataframe().to_csv(
                        os.path.join(config["save_path"], "optuna_trials.csv")
                    )
                ],
            )
            print_study_results(study)
        else:
            with tf.Graph().as_default() as graph:
                with tf.Session(
                    config=tf.ConfigProto(
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True),
                    )
                ) as sess:
                    train(args, sess, graph, config)
    elif args.job == "eval":
        with tf.Graph().as_default() as graph:
            with tf.Session(
                config=tf.ConfigProto(
                    log_device_placement=False,
                    gpu_options=tf.GPUOptions(allow_growth=True),
                )
            ) as sess:
                evaluate(args, sess, graph, config)


if __name__ == "__main__":
    main()

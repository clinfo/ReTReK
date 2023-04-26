import tensorflow as tf
import numpy as np
from functools import partial


def _objective(trial, train_fn, tune_params, args, config):
    """
    Objective function optimized during hyper parameter search
    parameters to tune must be a list of following objects:
    ["cat", name, [value1, value2,...]]
    ["float", name, start, stop, log or step]
    ["int", name, start, stop, step]

    Args:
        trial (optuna.trial.Trial): trial instance from optuna
        tune_params (list): list of parameters to tune
        args (dict): Settings
        config (dict): Training settings
    Returns:
        val_score (float): Score of trial on validation set
    """

    def add_param(param_tuple, config):
        param_type, param = param_tuple[0], param_tuple[1]
        if param_type == "cat":
            config[param] = trial.suggest_categorical(param, param_tuple[2])
        elif param_type == "float":
            low, high = param_tuple[2], param_tuple[3]
            if type(param_tuple[4]) == bool:
                config[param] = trial.suggest_float(
                    param, low, high, log=param_tuple[4]
                )
            else:
                config[param] = trial.suggest_float(
                    param, low, high, step=param_tuple[4]
                )
        elif param_type == "int":
            low, high, step = param_tuple[2], param_tuple[3], param_tuple[4]
            config[param] = trial.suggest_int(param, low, high, step=step)

    for param_tuple in tune_params:
        add_param(param_tuple, config)

    with tf.Graph().as_default() as graph:
        with tf.Session(
            config=tf.ConfigProto(
                log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True)
            )
        ) as sess:
            val_score = train_fn(args, sess, graph, config, return_score=True)
    return val_score


def get_objective(train_fn, tune_params, args, config):
    """
    Args:
        train_fn (Callable): Function used to train a model
        tune_params (list): list of hyper parameters to tune
        args (argsparse.Namespace): arguments to be passed to the train function.
        config (dict): dictionnary containing meta information on training
    Returns:
        objective (Callable): objective function for hyper parameter optimization
    """
    return partial(
        _objective, train_fn=train_fn, tune_params=tune_params, args=args, config=config
    )


def print_study_results(study):
    """
    Args:
        study (optuna.study.Study): Optuna study
    """

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def metrics_to_score(metrics, config):
    """
    Aggregates metrics into a single score to maximize during hyper parameter optimization
    Args:
        metrics (dict): dictionnary of metrics
        config (dict): dictionnary containing meta information on training
    Returns:
        (float): single score to optimize during hyper parameter optimization
    """
    return (
        metrics["accuracy"]
        + metrics["auc"]
        + config["regression_weight"] * np.mean(metrics["r2"])
        + config["category_weight"] * metrics["accuracy_step"]
    ) / (2 + config["regression_weight"] + config["category_weight"])

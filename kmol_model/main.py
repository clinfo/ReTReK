import argparse
import logging
import torch
from kmol.core.config import Config
from kmol.core.helpers import SuperFactory
from kmol.data.splitters import AbstractSplitter
from executors import *
from dataset import *
from gnn import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", choices=["train", "eval", "bayesian_opt"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-output-path", type=str, default="")

    args = parser.parse_args()
    return args


def to_loader(dataset, batch_size, shuffle, num_workers, follow_batch=None):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=NoSetDeviceCollater().apply,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def init_splits(dataset, config):
    splitter = SuperFactory.create(AbstractSplitter, config.splitter)
    splits = splitter.apply(data_loader=dataset)
    return splits


def init_loader(dataset, args, config, splits, split_name):
    subset = torch.utils.data.Subset(dataset=dataset, indices=splits[split_name])
    return to_loader(subset, config.batch_size, split_name=="train", args.num_workers)


if __name__ == "__main__":
    args = parse_args()
    config = Config.from_json(args.config)
    dataset = CSVDataset(cache_location=config.cache_location, **config.loader)
    splits = init_splits(dataset, config)

    if args.task =="train":
        train_loader = init_loader(dataset, args, config, splits, "train")
        val_loader = init_loader(dataset, args, config, splits, "validation")
        trainer = Trainer(config)
        trainer.run(train_loader, val_loader)
    elif args.task == "eval":
        test_loader = init_loader(dataset, args, config, splits, "test")
        if args.eval_output_path:
            config = config.cloned_update(output_path=args.eval_output_path)
        evaluator = Evaluator(config)
        evaluator.run(test_loader)
    elif args.task == "bayesian_opt":
        train_loader = init_loader(dataset, args, config, splits, "train")
        val_loader = init_loader(dataset, args, config, splits, "validation")
        executor = BayesianOptimizer(args.config)
        executor.run(train_loader, val_loader)
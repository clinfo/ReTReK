import argparse
import torch
from pathlib import Path
from kmol.core.config import Config
from executors import *
from dataset import *
from gnn import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-model", type=str, required=True, help="Path config file of classification model.")
    parser.add_argument("--regression-model", type=str, required=False, help="Path to config file of regression model.")
    parser.add_argument("--data", type=str, required=True, help="Path to file containing one smiles per line or single smiles string.")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--featurizer", type=str, choices=["graph", "ecfp", "mordred"], default="graph")

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


if __name__ == "__main__":
    args = parse_args()
    dataset = CSVDataset(input_path=None, target_columns=[], featurizer=args.featurizer, use_cache=False)
    if Path(args.data).exists():
        with Path(args.data).open("r") as f:
            smiles = [l.split("\n")[0] for l in f.readlines()]
        dataset.data = pd.DataFrame({"smiles": smiles})
    else:
        dataset.data = pd.DataFrame({"smiles": [args.data]})

    print(f"Number of smiles to process: {len(dataset)}")
    loader = to_loader(dataset, 64, False, 4)

    # Is solved prediction
    solved_preds = Predictor(config=Config.from_json(args.classification_model)).run(loader)[:, 0]
    solved_preds = torch.sigmoid(solved_preds) # converting logits to probability

    results = pd.DataFrame({"smiles": dataset.data.smiles.values, "solved": solved_preds})

    if args.regression_model is not None:
        # Keep only samples predicted to be solved for regression
        to_filter_out = solved_preds < 0.5
        reg_config = Config.from_json(args.regression_model)
        regression_preds = Predictor(config=reg_config).run(loader)
        for i, output in enumerate(reg_config.loader["target_columns"]):
            preds = regression_preds[:, i]
            preds[to_filter_out.tolist()] = np.nan
            results[output] = preds


    print(results)
    results.to_csv(args.save_path, index=False)
    print(f"Results saved to : {args.save_path}")

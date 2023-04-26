import numpy as np
import pandas as pd
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
from typing import NamedTuple



class TemplateStat(NamedTuple):
    minimum: int
    maximum: int
    average: int


class Reader:
    def __init__(self, uspto_data_path, template_column, filter_multi_part_core: bool = True):
        templates = pd.read_csv(uspto_data_path)
        if filter_multi_part_core:
            templates["multi_part_core"] = templates[template_column].apply(lambda x: "." in x.split(">>")[-1])
            templates = templates[~templates["multi_part_core"]]
        self.template_counts = templates[template_column].value_counts()

    @staticmethod
    def read_smiles(compound_folder):
        f = (compound_folder / "state.sma").open("r")
        return f.read().split("\n")[-1]
    
    @staticmethod
    def read_solved(compound_folder):
        return int((compound_folder / "proven").exists())
    
    @staticmethod
    def read_time(compound_folder):
        f = (compound_folder / "time.txt").open("r")
        return float(f.read())
    
    @staticmethod
    def read_steps(compound_folder):
        f = (compound_folder / "state.sma").open("r")
        return len(f.read().split("\n")) - 1
    
    def get_template_stats(self, compound_folder):
        f = (compound_folder / "reaction.sma").open("r")
        reactions = f.read().split("\n")
        if reactions[0] != '':
            counts = self.template_counts[reactions].values
        else:
            counts = np.array([0])
        return TemplateStat(
            minimum=counts.min(),
            maximum=counts.max(),
            average=int(counts.mean())
        )
    
    def read_compound(self, compound_folder):
        smiles = Reader.read_smiles(compound_folder)
        solved = Reader.read_solved(compound_folder)
        time = Reader.read_time(compound_folder)
        if solved:
            steps = Reader.read_steps(compound_folder)
            template_counts_stats = self.get_template_stats(compound_folder)
        else:
            steps = np.nan
            template_counts_stats = TemplateStat(
                minimum=np.nan,
                maximum=np.nan,
                average=np.nan
            )
        return {
            "smiles": smiles,
            "solved": solved,
            "time": time, 
            "steps": steps,
            "template_min_count": template_counts_stats.minimum,
            "template_max_count": template_counts_stats.maximum,
            "template_mean_count": template_counts_stats.average,
        }
    
    def extract(self, results_folder, add_log_transforms: bool = True):
        if type(results_folder) == str:
            results_folder = Path(results_folder)
        data = []
        for compound_folder in tqdm(results_folder.glob("*")):
            sample = self.read_compound(compound_folder)
            data.append(sample)
        df = pd.DataFrame(data)
        if add_log_transforms:
            df["log_time"] = np.log(df.time)
            df["log_template_min_count"] = np.log(df.template_min_count)
            df["log_template_max_count"] = np.log(df.template_max_count)
            df["log_template_mean_count"] = np.log(df.template_mean_count)
        return df


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--uspto-template-path", type=str, required=True)
    parser.add_argument("--template-column", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    reader = Reader(args.uspto_template_path, args.template_column)
    df = reader.extract(args.data_folder)
    df.to_csv(args.output_path, index=False)

import os
from data import *
from rdkit import Chem
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter


def get_templates_stats(templates, template_to_record):
    """
    Fetches statistics for a given list of templates:
    Args:
        templates (list str): list of reaction templates
        template_to_record (dict): dictionnary mapping a template to the number of records associated in a reference dataset
    Returns:
        Tuple (int) minimum, average and maximum number of associated records for the list of templates.
    """
    records_per_templates = [template_to_record[templ] for templ in templates]
    if 0 in records_per_templates and templates[0] != "":
        print("At least one template could not be found: ", templates)
    return (
        int(np.min(records_per_templates)),
        int(np.mean(records_per_templates)),
        int(np.max(records_per_templates)),
    )


def create_dataset(
    results_dir,
    mol_data_path,
    atom_num_limit,
    save_path,
    template_to_record,
    with_edge_features=False,
):
    """
    Creates a dataset to train or evaluate a kGCN model on and saves it as a json file.
    Args:
        results_dir (str): path to directory containing RealRetro results
        mol_data_path (str): path to directory containing the .mol files
        atom_num_limit (int): maximum number of atoms per molecule (molecules with more atoms will be ignored)
        save_path (str): path to save the dataset to.
        template_to_record (str): path to file mapping reaction templates to number of associated records.
        with_edge_features (bool): if set to True, edge features will be computed for each node.
            The per node edge features are computed as the sum of the edge features of the edges a node belongs to.
    """
    data = defaultdict(list)
    template_to_record = Counter(open(template_to_record, "r").read().split("\n"))
    for result_dir in tqdm(os.listdir(results_dir)):
        try:
            parameters = json.load(
                open(os.path.join(results_dir, result_dir, "parameters.json"), "r")
            )
            mol_file = parameters["target"].split("/")[-1][:-1]
            mol_path = os.path.join(mol_data_path, mol_file)
            mol = Chem.MolFromMolFile(mol_path)
            if mol.GetNumAtoms() > atom_num_limit:
                continue
            feature, adj, enabled_nodes, edge_feature = extract_mol_features(
                mol, atom_num_limit, with_edge_features
            )
            label = int("proven" in os.listdir(os.path.join(results_dir, result_dir)))
            time = float(
                open(os.path.join(results_dir, result_dir, "time.txt"), "r").read()
            )
            templates = (
                open(os.path.join(results_dir, result_dir, "reaction.sma"), "r")
                .read()
                .split("\n")
            )
            (min_rec, mean_rec, max_rec) = (
                get_templates_stats(
                    templates, template_to_record
                )
                if label
                else (0, 0, 0)
            )
            steps = (
                max(
                    pd.read_csv(
                        os.path.join(results_dir, result_dir, "best_tree_info.csv"),
                        sep="\t",
                    )["depth"]
                )
                if label
                else 0
            )
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            print(
                f"Something went wrong processing {os.path.join(results_dir,result_dir)}"
            )
            print(f"Error message : {e}")
            continue

        data["features"].append(feature)
        data["paths"].append(parameters["target"][1:-1])
        data["adjs"].append([a.tolist() for a in adj])
        data["edges"].append(edge_feature)
        data["labels"].append([label])
        data["time"].append([time])
        data["steps"].append([steps])
        data["min_recs"].append([min_rec])
        data["mean_recs"].append([mean_rec])
        data["max_recs"].append([max_rec])
        data["enabled_nodes"].append([enabled_nodes])
        data["smiles"].append([smiles])

    with open(save_path, "w") as f:
        json.dump(data, f)


def get_parser():
    """
    Parse arguments
    Returns:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="description", usage="usage")

    parser.add_argument(
        "-r",
        "--result-dir",
        required=True,
        type=str,
        help="Path to RealRetro results directory",
    )

    parser.add_argument(
        "-d", "--data-dir", required=True, type=str, help="Path to mol files directory"
    )

    parser.add_argument(
        "-n",
        "--n-atom-limit",
        required=False,
        type=int,
        default=50,
        help="Atom num limit",
    )

    parser.add_argument(
        "-s",
        "--save-path",
        required=True,
        type=str,
        default="",
        help="Save path for created dataset",
    )

    parser.add_argument(
        "-tm",
        "--template-mapper",
        required=False,
        type=str,
        default="../data/templates_mapped.sma",
        help="Path to csv file mapping each USPTO reaction to a template reaction",
    )

    parser.add_argument(
        "-we",
        "--with-edges",
        action="store_true",
        default=False,
        help="Also computes edges features (per node)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    create_dataset(
        args.result_dir,
        args.data_dir,
        args.n_atom_limit,
        args.save_path,
        args.template_mapper,
        args.with_edges,
    )

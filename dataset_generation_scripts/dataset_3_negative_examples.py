""" Generate the positive and negative chemical reaction examples. """

import argparse
import os
import random

import multiprocessing as mp
import numpy as np
import pandas as pd

from tqdm import tqdm

from rdkit import RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")


def get_parser():
    """ Parse user-specified arguments. """

    arg_parser = argparse.ArgumentParser(
        description="description",
        usage="usage"
    )

    arg_parser.add_argument(
        "-iai", "--input_ai_file_path", required=True, type=str, help="Path of the additional information dataset file."
    )

    arg_parser.add_argument(
        "-irt", "--input_rt_file_path", required=True, type=str, help="Path of the reaction template dataset file."
    )

    arg_parser.add_argument(
        "-o", "--output_folder_path", required=True, type=str, help="Path of the folder where the outputs are stored."
    )

    arg_parser.add_argument(
        "-y_conf", "--yield_confidence", required=False, type=str, default="high",
        help="The reaction yield confidence value."
    )

    arg_parser.add_argument(
        "-y_cutoff", "--yield_cutoff", required=False, type=int, default=20,
        help="The reaction yield percentage cutoff value."
    )

    arg_parser.add_argument(
        "-min_tf", "--min_template_frequency", required=False, type=int, default=1,
        help="The minimum reaction template frequency."
    )

    arg_parser.add_argument(
        "-rt_vn", "--rt_virtual_negatives", required=False, type=int, default=3,
        help="The number of virtual negative examples generated using random reaction templates."
    )

    arg_parser.add_argument(
        "-rp_vn", "--rp_virtual_negatives", required=False, type=int, default=7,
        help="The number of virtual negative examples generated using random perturbations."
    )

    arg_parser.add_argument(
        "-seed", "--random_seed", required=False, type=int, default=101, help="Random seed value."
    )

    arg_parser.add_argument(
        "-cores", "--num_cores", required=False, type=int, default=1, help="Number of CPU cores for multiprocessing."
    )

    return arg_parser.parse_args()


args = get_parser()

random.seed(args.random_seed)
np.random.seed(args.random_seed)

negative_examples_folder_path = os.path.join(args.output_folder_path, "dataset_3_negative_examples")

if not os.path.exists(negative_examples_folder_path):
    os.mkdir(negative_examples_folder_path)


print("(1/4) Reading and preparing datasets...", end="", flush=True)

additional_information_df = pd.read_pickle(args.input_ai_file_path)[
    ["publication_year", "document_id", "reaction_smiles", "reaction_yield"]
]
reaction_template_df = pd.read_csv(args.input_rt_file_path, low_memory=False)[
    ["publication_year", "patent_id", "original_reaction_smiles", "new_reaction_smiles", "forward_template",
     "main_product", "template_count"]
]

reaction_template_df = reaction_template_df[reaction_template_df["template_count"] >= args.min_template_frequency]

reaction_yield_lookup = {
    (int(val_tuple[0]), val_tuple[1], val_tuple[2]): val_tuple[3]
    for val_tuple in additional_information_df.values.tolist()
}

print("done.\n")


reaction_yield_percentage_values = []

for _, row in tqdm(reaction_template_df.iterrows(), total=len(reaction_template_df.index), ascii=True, ncols=120,
                   desc="(2/4) Analyzing reaction yield information"):
    row_key = (row["publication_year"], row["patent_id"], row["original_reaction_smiles"])

    all_percentage_values = []

    if row_key in reaction_yield_lookup:
        if row["main_product"] in reaction_yield_lookup[row_key]:
            for yield_entries in reaction_yield_lookup[row_key][row["main_product"]]:
                if "percent" in yield_entries[0] and yield_entries[4] == args.yield_confidence.lower():
                    all_percentage_values.append(yield_entries[1])
                    all_percentage_values.append(yield_entries[3])

    if len([x for x in all_percentage_values if x is not None and 0 <= x <= 100]) > 0:
        reaction_yield_percentage_values.append(
            np.array([x for x in all_percentage_values if x is not None and 0 <= x <= 100]).mean()
        )
    else:
        reaction_yield_percentage_values.append(None)

reaction_template_df["label"] = reaction_yield_percentage_values
reaction_template_df.dropna(subset=["label"])
reaction_template_df["label"] = [1 if x > 20 else 0 for x in reaction_template_df["label"].values.tolist()]

unique_reaction_templates = list(set(reaction_template_df["forward_template"].values.tolist()))
unique_reaction_products = list(set(reaction_template_df["main_product"].values.tolist()))


print("")


def generate_virtual_negative_examples(df_row):
    """ Generate the virtual negative reaction examples. """

    reactant_mols = []

    for reactant in df_row["new_reaction_smiles"].split(">>")[0].split("."):
        reactant_mol = AllChem.MolFromSmiles(reactant)

        for atom in reactant_mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")

        AllChem.SanitizeMol(reactant_mol)
        reactant_mols.append(reactant_mol)

    rt_virtual_negative_products = set()

    for forward_template in random.sample(unique_reaction_templates, len(unique_reaction_templates)):
        if df_row["forward_template"] != forward_template and \
                len(forward_template.split(">>")[0].split(".")) == len(reactant_mols):
            try:
                reaction = AllChem.ReactionFromSmarts(forward_template)

                AllChem.SanitizeRxn(reaction)
                reaction.Initialize()

                reaction_outcomes = reaction.RunReactants(reactant_mols, maxProducts=5)

                for reaction_outcome in reaction_outcomes:
                    if len(reaction_outcome) == 1:
                        try:
                            AllChem.SanitizeMol(reaction_outcome[0])

                            negative_product_smiles = AllChem.MolToSmiles(reaction_outcome[0], canonical=True)

                            if negative_product_smiles != df_row["main_product"]:
                                rt_virtual_negative_products.add(negative_product_smiles)

                        except:
                            continue

            except:
                continue

        if len(rt_virtual_negative_products) >= args.rt_virtual_negatives:
            if len(rt_virtual_negative_products) > args.rt_virtual_negatives:
                rt_virtual_negative_products = random.sample(
                    list(rt_virtual_negative_products), args.rt_virtual_negatives
                )

            break

    rp_virtual_negative_products = set()

    for reaction_product in random.sample(unique_reaction_products, len(unique_reaction_products)):
        if df_row["main_product"] != reaction_product:
            rp_virtual_negative_products.add(reaction_product)

        if len(rp_virtual_negative_products) >= args.rp_virtual_negatives:
            break

    reactant_side = ".".join([AllChem.MolToSmiles(reactant_mol, canonical=True) for reactant_mol in reactant_mols])

    return [reactant_side + ">>" + rtvn_product for rtvn_product in rt_virtual_negative_products], \
           [reactant_side + ">>" + rpvn_product for rpvn_product in rp_virtual_negative_products]


with mp.Pool(args.num_cores) as process_pool:
    processing_results = [processed_entry for processed_entry in tqdm(
        iterable=process_pool.imap(generate_virtual_negative_examples, [
            row for _, row in reaction_template_df.iterrows()
        ]), total=len(reaction_template_df.index), ascii=True, ncols=120,
        desc=f"(3/4) Generating the negative reaction examples"
    )]

    process_pool.close()
    process_pool.join()


print("\n(4/4) Finalizing the in-scope filter datasets...", end="", flush=True)

ptn_in_scope_df = reaction_template_df[["new_reaction_smiles", "main_product", "label"]]
ptn_in_scope_df.columns = ["reaction_smiles", "main_product", "label"]

rt_virtual_negatives, rp_virtual_negatives = [], []

for processed_entry in processing_results:
    rt_virtual_negatives.extend(processed_entry[0])
    rp_virtual_negatives.extend(processed_entry[1])

pvn_in_scope_df = pd.concat([
    ptn_in_scope_df[ptn_in_scope_df["label"] == 1],
    pd.DataFrame({
        "reaction_smiles": rt_virtual_negatives,
        "main_product": [reaction_smiles.split(">>")[1] for reaction_smiles in rt_virtual_negatives],
        "label": [0 for _ in rt_virtual_negatives]
    }),
    pd.DataFrame({
        "reaction_smiles": rp_virtual_negatives,
        "main_product": [reaction_smiles.split(">>")[1] for reaction_smiles in rp_virtual_negatives],
        "label": [0 for _ in rp_virtual_negatives]
    })
])

ptnvn_in_scope_df = pd.concat([
    ptn_in_scope_df,
    pd.DataFrame({
        "reaction_smiles": rt_virtual_negatives,
        "main_product": [reaction_smiles.split(">>")[1] for reaction_smiles in rt_virtual_negatives],
        "label": [0 for _ in rt_virtual_negatives]
    }),
    pd.DataFrame({
        "reaction_smiles": rp_virtual_negatives,
        "main_product": [reaction_smiles.split(">>")[1] for reaction_smiles in rp_virtual_negatives],
        "label": [0 for _ in rp_virtual_negatives]
    })
])

print("done.")

ptn_in_scope_df.to_csv(os.path.join(negative_examples_folder_path, "inscope_filter_dataset_PTN.csv"), index=False)
pvn_in_scope_df.to_csv(os.path.join(negative_examples_folder_path, "inscope_filter_dataset_PVN.csv"), index=False)
ptnvn_in_scope_df.to_csv(os.path.join(negative_examples_folder_path, "inscope_filter_dataset_PTNVN.csv"), index=False)

print(f"\nNumber of PTN dataset entries: {len(ptn_in_scope_df.index)}")
print(f"Number of PVN dataset entries: {len(pvn_in_scope_df.index)}")
print(f"Number of PTNVN dataset entries: {len(ptnvn_in_scope_df.index)}")

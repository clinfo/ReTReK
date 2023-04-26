""" Generate the global evaluation datasets. """

import argparse
import os
import random

import multiprocessing as mp
import pandas as pd

from itertools import chain
from functools import reduce
from math import ceil
from tqdm import tqdm

from rdkit import RDLogger

from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Normalize

from rdkit.DataStructs import BulkTanimotoSimilarity

from rdkit.ML.Cluster import Butina


def get_parser():
    """ Parse user-specified arguments. """

    arg_parser = argparse.ArgumentParser(
        description="description",
        usage="usage"
    )

    arg_parser.add_argument(
        "-i", "--input_file_path", required=True, type=str, help="Path of any raw ChEMBL .txt file."
    )

    arg_parser.add_argument(
        "-o", "--output_folder_path", required=True, type=str, help="Path of the folder where the outputs are stored."
    )

    arg_parser.add_argument(
        "-min_at", "--min_atoms", required=False, type=int, default=10, help="The minimum atoms cutoff value."
    )

    arg_parser.add_argument(
        "-max_at", "--max_atoms", required=False, type=int, default=50, help="The maximum atoms cutoff value."
    )

    arg_parser.add_argument(
        "-fp_cutoff", "--fp_similarity_cutoff", required=False, type=float, default=0.2,
        help="The fingerprint similarity cutoff value during Butina clustering."
    )

    arg_parser.add_argument(
        "-sm_size", "--small_dataset_size", required=False, type=int, default=1000,
        help="The size of the small evaluation dataset."
    )

    arg_parser.add_argument(
        "-lg_size", "--large_dataset_size", required=False, type=int, default=27000,
        help="The size of the large evaluation dataset."
    )

    arg_parser.add_argument(
        "-out_pct", "--outlier_percentage", required=False, type=float, default=0.05,
        help="The percentage of outliers included in the final evaluation datasets."
    )

    arg_parser.add_argument(
        "-seed", "--random_seed", required=False, type=int, default=101, help="Random seed value."
    )

    arg_parser.add_argument(
        "-cores", "--num_cores", required=False, type=int, default=1, help="Number of CPU cores for multiprocessing."
    )

    return arg_parser.parse_args()


def standardize_chembl_entry(smiles_string):
    """ Standardize a single chemical compound entry from the ChEMBL dataset. """

    try:
        mol = AllChem.MolFromSmiles(smiles_string)
        AllChem.SanitizeMol(mol)

        salt_remover = SaltRemover(defnFilename="./all_salts.txt")
        mol = salt_remover.StripMol(mol)

        if mol is None:
            return None

        elif mol.GetNumAtoms() == 0:
            return None

        else:
            smiles_string = AllChem.MolToSmiles(mol, canonical=True)

            if "." in smiles_string:
                smiles_string = sorted(smiles_string.split("."), key=len, reverse=True)[0]

            mol = AllChem.MolFromSmiles(smiles_string)
            AllChem.SanitizeMol(mol)

            mol = Normalize(mol)

            return mol

    except:
        return None


def run_butina_clustering(mol_fingerprints, cutoff_value):
    """ Run the Butina Clustering algorithm on a collection of fingerprint descriptors. """

    clustering_distances = []

    for i in range(1, len(mol_fingerprints)):
        similarity_values = BulkTanimotoSimilarity(mol_fingerprints[i], mol_fingerprints[:i])
        clustering_distances.extend([1 - similarity_value for similarity_value in similarity_values])

    butina_clusters = Butina.ClusterData(clustering_distances, len(mol_fingerprints), cutoff_value, isDistData=True)

    return butina_clusters


def sample_representative_compounds(representative_compound_clusters, num_samples):
    """ Sample an appropriate amount of representative compounds from each cluster. """

    if num_samples >= reduce(lambda count, k: count + len(k), representative_compound_clusters, 0):
        return list(chain.from_iterable(representative_compound_clusters))

    else:
        sampled_representative_compounds = []
        samples_per_cluster = ceil(num_samples / len(representative_compound_clusters))

        for compound_cluster_ind, compound_cluster in enumerate(representative_compound_clusters):
            if samples_per_cluster <= len(compound_cluster):
                sampled_representative_compounds.extend(random.sample(compound_cluster, samples_per_cluster))

            else:
                sampled_representative_compounds.extend(random.sample(compound_cluster, len(compound_cluster)))

            if len(representative_compound_clusters) - (compound_cluster_ind + 1) != 0:
                samples_per_cluster = ceil(
                    (num_samples - len(sampled_representative_compounds)) /
                    (len(representative_compound_clusters) - (compound_cluster_ind + 1))
                )

                if samples_per_cluster <= 0:
                    samples_per_cluster = 1

        return random.sample(sampled_representative_compounds, num_samples)


def main():
    """ The main function of the script. """

    RDLogger.DisableLog("rdApp.*")

    args = get_parser()

    random.seed(args.random_seed)

    global_evaluation_folder_path = os.path.join(args.output_folder_path, "dataset_1_global_evaluation")

    if not os.path.exists(global_evaluation_folder_path):
        os.mkdir(global_evaluation_folder_path)

    chembl_df = pd.read_csv(args.input_file_path, sep="\t", header=0)
    print(f"Number of entries in the original ChEMBL dataset: {len(chembl_df.index)}\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 1: Standardize all ChEMBL dataset entries.
    # ------------------------------------------------------------------------------------------------------------------
    chembl_df = chembl_df.dropna(subset=["canonical_smiles"])
    chembl_df = chembl_df.drop_duplicates(subset=["canonical_smiles"])["canonical_smiles"]

    with mp.Pool(args.num_cores) as process_pool:
        processed_chembl_entries = [
            processed_entry for processed_entry in tqdm(process_pool.imap(standardize_chembl_entry, chembl_df.values),
                                                        total=len(chembl_df.index),
                                                        ascii=True,
                                                        ncols=120,
                                                        desc="(1/3) Standardizing ChEMBL dataset entries")
            if processed_entry is not None and args.min_atoms <= processed_entry.GetNumAtoms() <= args.max_atoms
        ]

        process_pool.close()
        process_pool.join()

    processed_chembl_entries.sort(key=lambda mol: mol.GetNumAtoms())

    print(f"\nNumber of ChEMBL dataset entries after standardization: {len(processed_chembl_entries)}\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 2: Run partial Butina clustering on the standardized ChEMBL dataset entries.
    # ------------------------------------------------------------------------------------------------------------------
    outlier_compound_indices, representative_compound_clusters, partition_size = [], [], 10000

    for partition_index in tqdm(range(0, int(len(processed_chembl_entries) / partition_size) + 1), ascii=True,
                                ncols=120, desc="(2/3) Running the partial Butina clustering algorithm"):
        if (partition_index + 1) * partition_size < len(processed_chembl_entries):
            partition_fingerprints = [
                AllChem.GetMorganFingerprintAsBitVect(entry, radius=2, nBits=1024) for entry
                in processed_chembl_entries[partition_index * partition_size:(partition_index + 1) * partition_size]
            ]

        else:
            partition_fingerprints = [
                AllChem.GetMorganFingerprintAsBitVect(entry, radius=2, nBits=1024) for entry
                in processed_chembl_entries[partition_index * partition_size:len(processed_chembl_entries)]
            ]

        butina_clusters = run_butina_clustering(partition_fingerprints, cutoff_value=args.fp_similarity_cutoff)

        for butina_cluster in butina_clusters:
            if len(butina_cluster) == 1:
                outlier_compound_indices.extend(
                    [compound_index + (partition_index * partition_size) for compound_index in butina_cluster]
                )

            else:
                representative_compound_clusters.append(
                    [compound_index + (partition_index * partition_size) for compound_index in butina_cluster]
                )

    outlier_compound_indices.sort()
    representative_compound_clusters.sort(key=len)

    total_outlier_compounds = len(outlier_compound_indices)
    total_representative_clusters = len(representative_compound_clusters)
    total_representative_compounds = reduce(lambda count, k: count + len(k), representative_compound_clusters, 0)

    print(f"\nNumber of generated outliers after partial clustering: {total_outlier_compounds}")
    print(f"Number of generated representative clusters after partial clustering: {total_representative_clusters}")
    print(f"Number of generated representatives after partial clustering: {total_representative_compounds}")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 3: Generate the evaluation datasets.
    # ------------------------------------------------------------------------------------------------------------------
    small_evaluation_dataset, large_evaluation_dataset = [], []

    number_of_outliers_small = total_outlier_compounds \
        if int(args.small_dataset_size * args.outlier_percentage) >= total_outlier_compounds \
        else int(args.small_dataset_size * args.outlier_percentage)

    small_evaluation_dataset.extend(random.sample(outlier_compound_indices, number_of_outliers_small))

    number_of_representatives_small = total_representative_compounds \
        if (args.small_dataset_size - number_of_outliers_small) >= total_representative_compounds \
        else (args.small_dataset_size - number_of_outliers_small)

    small_evaluation_dataset.extend(sample_representative_compounds(
        representative_compound_clusters, number_of_representatives_small
    ))

    print("\nNumber of small evaluation dataset entries and the representative to outlier ratio: "
          f"{number_of_representatives_small + number_of_outliers_small} "
          f"({number_of_representatives_small}:{number_of_outliers_small})")

    number_of_outliers_large = total_outlier_compounds \
        if int(args.large_dataset_size * args.outlier_percentage) >= total_outlier_compounds \
        else int(args.large_dataset_size * args.outlier_percentage)

    large_evaluation_dataset.extend(random.sample(outlier_compound_indices, number_of_outliers_large))

    number_of_representatives_large = total_representative_compounds \
        if (args.large_dataset_size - number_of_outliers_large) >= total_representative_compounds \
        else (args.large_dataset_size - number_of_outliers_large)

    large_evaluation_dataset.extend(sample_representative_compounds(
        representative_compound_clusters, number_of_representatives_large
    ))

    print("Number of large evaluation dataset entries and the representative to outlier ratio: "
          f"{number_of_representatives_large + number_of_outliers_large} "
          f"({number_of_representatives_large}:{number_of_outliers_large})\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 4: Get the SMILES string representations for the indices in each of the evaluation datasets.
    # ------------------------------------------------------------------------------------------------------------------
    small_evaluation_dataset_smiles, large_evaluation_dataset_smiles = [], []

    print("(3/3) Retrieving SMILES strings for indices and saving the evaluation datasets...", end="", flush=True)

    for small_evaluation_dataset_index in small_evaluation_dataset:
        small_evaluation_dataset_smiles.append(
            AllChem.MolToSmiles(processed_chembl_entries[small_evaluation_dataset_index], canonical=True)
        )

    for large_evaluation_dataset_index in large_evaluation_dataset:
        large_evaluation_dataset_smiles.append(
            AllChem.MolToSmiles(processed_chembl_entries[large_evaluation_dataset_index], canonical=True)
        )

    random.shuffle(small_evaluation_dataset_smiles)
    random.shuffle(large_evaluation_dataset_smiles)

    pd.DataFrame({"canonical_smiles": small_evaluation_dataset_smiles}, dtype=str).to_csv(
        os.path.join(global_evaluation_folder_path, "small_evaluation_dataset.csv"), index=False
    )

    pd.DataFrame({"canonical_smiles": large_evaluation_dataset_smiles}, dtype=str).to_csv(
        os.path.join(global_evaluation_folder_path, "large_evaluation_dataset.csv"), index=False
    )

    print("done.")


if __name__ == "__main__":
    main()

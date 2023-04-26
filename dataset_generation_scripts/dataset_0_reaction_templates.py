""" Generate the global chemical reaction datasets. """

import argparse
import csv
import multiprocessing
import os
import random
import re
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

from rdchiral.template_extractor import extract_from_reaction


def get_parser():
    """ Parse user-specified arguments. """

    arg_parser = argparse.ArgumentParser(
        description="description",
        usage="usage"
    )

    arg_parser.add_argument(
        "-i", "--input_file_path", required=True, type=str, help="Path of the USPTO Grants or Applications .rsmi file."
    )

    arg_parser.add_argument(
        "-o", "--output_folder_path", required=True, type=str, help="Path of the folder where the outputs are stored."
    )

    arg_parser.add_argument(
        "-seed", "--random_seed", required=False, type=int, default=101, help="Random seed value."
    )

    arg_parser.add_argument(
        "-cores", "--num_cores", required=False, type=int, default=1, help="Number of CPU cores for multiprocessing."
    )

    return arg_parser.parse_args()


def get_rxn_smiles(prod, reactants):
    """ The original GLN get_rxn_smiles function code. """

    prod_smi = Chem.MolToSmiles(prod, True)

    # Get rid of reactants when they don't contribute to this prod
    prod_maps = set(re.findall('\:([[0-9]+)\]', prod_smi))
    reactants_smi_list = []

    for mol in reactants:
        if mol is None:
            continue

        used = False

        for a in mol.GetAtoms():
            if a.HasProp('molAtomMapNumber'):
                if a.GetProp('molAtomMapNumber') in prod_maps:
                    used = True
                else:
                    a.ClearProp('molAtomMapNumber')
        if used:
            reactants_smi_list.append(Chem.MolToSmiles(mol, True))

    reactants_smi = '.'.join(reactants_smi_list)

    return '{}>>{}'.format(reactants_smi, prod_smi)


def get_writer(fname, header):
    """ The original GLN get_writer function code. """

    output_name = os.path.join('/data/hhasic/gln_test/', fname)
    fout = open(output_name, 'w')
    writer = csv.writer(fout)
    writer.writerow(header)

    return fout, writer


def get_tpl(task):
    """ The original GLN get_tpl function code. """

    idx, row_idx, rxn_smiles = task
    react, reagent, prod = rxn_smiles.split('>')
    reaction = {'_id': row_idx, 'reactants': react, 'products': prod}

    try:
        template = extract_from_reaction(reaction)
    except:
        return idx, {'err_msg': "exception"}

    return idx, template


def gln_clean_uspto(uspto_rsmi_file_path: str, save_folder_path: str):
    """ The original GLN clean_uspto script code. """

    uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"
    fname = uspto_rsmi_file_path

    # ------------------------------------------------------------------------------------------------------------------

    seed = 19260817
    np.random.seed(seed)
    random.seed(seed)

    split_mode = 'multi'  # single or multi

    pt = re.compile(r':(\d+)]')
    cnt = 0
    clean_list = []
    set_rxn = set()
    num_single = 0
    num_multi = 0
    bad_mapping = 0
    bad_prod = 0
    missing_map = 0
    raw_num = 0

    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        pbar = tqdm(reader)
        bad_rxn = 0
        for row in pbar:
            rxn_smiles = row[header.index('ReactionSmiles')]
            all_reactants, reagents, prods = rxn_smiles.split('>')
            all_reactants = all_reactants.split()[0]  # remove ' |f:1...'
            prods = prods.split()[0]  # remove ' |f:1...'
            if '.' in prods:
                num_multi += 1
            else:
                num_single += 1
            if split_mode == 'single' and '.' in prods:  # multiple prods
                continue
            rids = ','.join(sorted(re.findall(pt, all_reactants)))
            pids = ','.join(sorted(re.findall(pt, prods)))
            if rids != pids:  # mapping is not 1:1
                bad_mapping += 1
                continue
            reactants = [Chem.MolFromSmiles(smi) for smi in all_reactants.split('.')]

            for sub_prod in prods.split('.'):
                mol_prod = Chem.MolFromSmiles(sub_prod)
                if mol_prod is None:  # rdkit is not able to parse the product
                    bad_prod += 1
                    continue
                # Make sure all have atom mapping
                if not all([a.HasProp('molAtomMapNumber') for a in mol_prod.GetAtoms()]):
                    missing_map += 1
                    continue

                raw_num += 1
                rxn_smiles = get_rxn_smiles(mol_prod, reactants)
                if not rxn_smiles in set_rxn:
                    clean_list.append((
                        row[header.index('PatentNumber')],
                        rxn_smiles,
                        row[header.index('ReactionSmiles')],
                        row[header.index('ParagraphNum')],
                        row[header.index('Year')]
                    ))
                    set_rxn.add(rxn_smiles)
            pbar.set_description('select: %d, dup: %d' % (len(clean_list), raw_num))

    print('# clean', len(clean_list))
    print('single', num_single, 'multi', num_multi)
    print('bad mapping', bad_mapping)
    print('bad prod', bad_prod)
    print('missing map', missing_map)
    print('raw extracted', raw_num)

    random.shuffle(clean_list)

    num_val = num_test = int(len(clean_list) * 0.1)

    for phase in ['val', 'test', 'train']:
        fout = os.path.join(save_folder_path,  f"uspto_{uspto_version}" + "_raw_%s.csv" % phase)

        with open(fout, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'document_id',
                'paragraph_num',
                'publication_year',
                'original_reaction_smiles',
                'reactants>reagents>production'
            ])

            if phase == 'val':
                r = range(num_val)
            elif phase == 'test':
                r = range(num_val, num_val + num_test)
            else:
                r = range(num_val + num_test, len(clean_list))
            for i in r:
                rxn_smiles = clean_list[i][1].split('>')
                result = []
                for r in rxn_smiles:
                    if len(r.strip()):
                        r = r.split()[0]
                    result.append(r)
                rxn_smiles = '>'.join(result)

                writer.writerow([
                    clean_list[i][0],
                    clean_list[i][3],
                    clean_list[i][4],
                    clean_list[i][2],
                    rxn_smiles
                ])


def gln_build_raw_template(uspto_rsmi_file_path: str, save_folder_path: str, num_cpu_cores: int):
    """ The original GLN build_raw_template script code. """

    uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"

    seed = 19260817
    np.random.seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------------------------------------------------------

    for dataset_split in ["train", "test", "val"]:
        fname = os.path.join(save_folder_path, f"uspto_{uspto_version}_raw_{dataset_split}.csv")

        with open(fname, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]

        pool = multiprocessing.Pool(num_cpu_cores)
        tasks = []
        for idx, row in tqdm(enumerate(rows)):
            # One of the indices always won't finish, thus exclude using the index from tqdm.
            if idx != 88303:
                row_idx, row_par_num, row_year, row_og_smi, rxn_smiles = row
                tasks.append((idx, row_idx, rxn_smiles))

        fout, writer = get_writer(
            os.path.join(save_folder_path, f"uspto_{uspto_version}_single_product_{dataset_split}.csv"),
            ['document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'new_reaction_smiles',
             'retro_templates']
        )

        fout_failed, failed_writer = get_writer(
            os.path.join(save_folder_path, f"uspto_{uspto_version}_failed_template_{dataset_split}.csv"),
            ['document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'rxn_smiles', 'err_msg']
        )

        for result in tqdm(pool.imap_unordered(get_tpl, tasks), total=len(tasks)):
            idx, template = result

            row_idx, row_par_num, row_year, row_og_smi, rxn_smiles = rows[idx]

            if 'reaction_smarts' in template:
                writer.writerow([row_idx, row_par_num, row_year, row_og_smi, rxn_smiles, template['reaction_smarts']])
                fout.flush()

                # print(f"finished {idx}")
            else:
                if 'err_msg' in template:
                    failed_writer.writerow(
                        [row_idx, row_par_num, row_year, row_og_smi, rxn_smiles, template['err_msg']]
                    )
                else:
                    failed_writer.writerow([row_idx, row_par_num, row_year, row_og_smi, rxn_smiles, "NaN"])

                fout_failed.flush()

                # print(f"finished {idx}")

        fout.close()
        fout_failed.close()


def post_process_reaction_templates(uspto_rsmi_file_path: str, save_folder_path: str):
    """ Post-process the cleaned and extracted USPTO reaction templates. """

    uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"

    train = pd.read_csv(os.path.join(save_folder_path, f"uspto_{uspto_version}_single_product_train.csv"))[
        ['document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'new_reaction_smiles',
         'retro_templates']
    ]
    validation = pd.read_csv(os.path.join(save_folder_path, f"uspto_{uspto_version}_single_product_val.csv"))[
        ['document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'new_reaction_smiles',
         'retro_templates']
    ]
    test = pd.read_csv(os.path.join(save_folder_path, f"uspto_{uspto_version}_single_product_test.csv"))[
        ['document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'new_reaction_smiles',
         'retro_templates']
    ]

    train.columns = ["patent_id", "paragraph_num", "publication_year", "original_reaction_smiles",
                     "new_reaction_smiles", "retro_template"]
    validation.columns = ["patent_id", "paragraph_num", "publication_year", "original_reaction_smiles",
                          "new_reaction_smiles", "retro_template"]
    test.columns = ["patent_id", "paragraph_num", "publication_year", "original_reaction_smiles", "new_reaction_smiles",
                    "retro_template"]

    clean_data = pd.concat([train, validation, test])

    clean_data["forward_template"] = [
        x.split(">>")[1] + ">>" + x.split(">>")[0] for x in clean_data["retro_template"].values
    ]

    clean_data["multi_part_core"] = [
        len(x.split(">")[0].split(".")) > 1 for x in clean_data["retro_template"].values
    ]

    clean_data = clean_data[[
        "patent_id", "paragraph_num", "publication_year", "original_reaction_smiles", "new_reaction_smiles",
        "forward_template", "retro_template", "multi_part_core"
    ]]

    unmapped_products, publication_year = [], []

    for data_tuple in tqdm(clean_data.values, total=len(clean_data.index), ascii=True, ncols=120,
                           desc="Post-processing extracted reaction templates"):
        try:
            mol = Chem.MolFromSmiles(data_tuple[4].split(">")[2])

            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

            Chem.SanitizeMol(mol)

            can_sm = Chem.MolToSmiles(mol, canonical=True)

            unmapped_products.append(can_sm)

        except:
            unmapped_products.append(None)

    clean_data["main_product"] = unmapped_products
    clean_data = clean_data.dropna(subset=["main_product"])
    clean_data = clean_data.drop_duplicates(subset=["main_product", "forward_template", "retro_template"])

    clean_data["template_count"] = clean_data.groupby("forward_template")["forward_template"].transform("count")

    product_template_lengths = []
    failed_ctr = 0

    for retro_template in clean_data["retro_template"].values:
        length = 0

        for core in retro_template.split(">")[2].split("."):
            try:
                length += len(Chem.MolFromSmarts(core).GetAtoms())
            except:
                failed_ctr += 1
                length += 1000

        product_template_lengths.append(length)

    clean_data["product_template_length"] = product_template_lengths

    clean_data = clean_data.sort_values(by=["publication_year"])

    clean_data = clean_data[["patent_id", "paragraph_num", "publication_year", "original_reaction_smiles",
                             "new_reaction_smiles", "forward_template", "retro_template", "main_product",
                             "multi_part_core", "product_template_length", "template_count"]]

    clean_data.to_csv(
        os.path.join(save_folder_path, f"uspto_{uspto_version}_reaction_templates_dataset.csv"), index=False
    )


def generate_kgcn_data(uspto_rsmi_file_path: str, save_folder_path: str, allow_multi_part_cores: bool = False,
                       min_frequency: int = None, max_product_template_length: int = None):
    """ Reads the dataset and generates the kGCN-ready dataset. """

    uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"

    source_data = pd.read_csv(
        os.path.join(save_folder_path, f"uspto_{uspto_version}_reaction_templates_dataset.csv"), low_memory=False
    )

    if not allow_multi_part_cores:
        source_data = source_data[~source_data["multi_part_core"]]

    if min_frequency is not None:
        source_data = source_data[source_data["template_count"] >= min_frequency]

    if max_product_template_length is not None:
        source_data = source_data[source_data["product_template_length"] <= max_product_template_length]

    kgcn_data = source_data[["main_product", "forward_template", "publication_year"]]
    kgcn_data.columns = ["product", "reaction_core", "max_publication_year"]

    kgcn_data.to_csv(os.path.join(save_folder_path, f"uspto_{uspto_version}_kgcn_dataset.csv"), index=False)


def clean_up_files(uspto_rsmi_file_path: str, save_folder_path: str):
    """ Clean up the additionally generated files. """

    uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"

    file_names_to_delete = [
        f"uspto_{uspto_version}_raw_train.csv",
        f"uspto_{uspto_version}_raw_val.csv",
        f"uspto_{uspto_version}_raw_test.csv",
        f"uspto_{uspto_version}_single_product_train.csv",
        f"uspto_{uspto_version}_single_product_val.csv",
        f"uspto_{uspto_version}_single_product_test.csv",
        f"uspto_{uspto_version}_failed_template_train.csv",
        f"uspto_{uspto_version}_failed_template_val.csv",
        f"uspto_{uspto_version}_failed_template_test.csv"
    ]

    for file_name in os.listdir(save_folder_path):
        if file_name in file_names_to_delete:
            os.remove(os.path.join(save_folder_path, file_name))


def main():
    """ The main function of the script. """

    warnings.simplefilter(action="ignore", category=FutureWarning)
    RDLogger.DisableLog("rdApp.*")

    args = get_parser()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    reaction_templates_folder_path = os.path.join(args.output_folder_path, "dataset_0_reaction_templates")

    if not os.path.exists(reaction_templates_folder_path):
        os.mkdir(reaction_templates_folder_path)

    print("(1/5) Running the GLN 'clean_uspto.py' script code:")
    gln_clean_uspto(args.input_file_path, reaction_templates_folder_path)

    print("\n(2/5) Running the GLN 'build_raw_template.py' script code:")
    gln_build_raw_template(args.input_file_path, reaction_templates_folder_path, args.num_cores)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    print("\n(3/5) Running the reaction template post-processing code:")
    post_process_reaction_templates(args.input_file_path, reaction_templates_folder_path)

    print("\n(4/5) Running the kGCN model dataset generation code:")
    generate_kgcn_data(args.input_file_path, reaction_templates_folder_path)

    print("\n(5/5) Running the clean up code:")
    clean_up_files(args.input_file_path, reaction_templates_folder_path)


if __name__ == "__main__":
    main()

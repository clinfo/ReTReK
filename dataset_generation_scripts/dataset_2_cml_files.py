""" Parse the USPTO dataset CML files and extract the additional information. """

import argparse
import os
import random

import multiprocessing as mp
import pandas as pd

from tqdm import tqdm
from quantulum3 import parser

from rdkit import RDLogger
from rdkit.Chem import AllChem

from xml.etree import ElementTree


def get_parser():
    """ Parse user-specified arguments. """

    arg_parser = argparse.ArgumentParser(
        description="description",
        usage="usage"
    )

    arg_parser.add_argument(
        "-i", "--input_folder_path", required=True, type=str, help="Path of the USPTO Grants or Applications folder."
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


def process_xml_reaction_entry(reaction_entry):
    """ Process a single chemical reaction entry from the USPTO dataset CML files. """

    cml_prefix = "{http://www.xml-cml.org/schema}"
    dl_prefix = "{http://bitbucket.org/dan2097}"

    # ------------------------------------------------------------------------------------------------------------------
    # Step 1: Extract the reaction entry source information and reaction SMILES string.
    # ------------------------------------------------------------------------------------------------------------------
    document_id = reaction_entry.find(f"{dl_prefix}source/{dl_prefix}documentId")
    document_id = document_id.text if document_id is not None else None

    heading_text = reaction_entry.find(f"{dl_prefix}source/{dl_prefix}headingText")
    heading_text = heading_text.text if heading_text is not None else None

    paragraph_num = reaction_entry.find(f"{dl_prefix}source/{dl_prefix}paragraphNum")
    paragraph_num = paragraph_num.text if paragraph_num is not None else None

    paragraph_text = reaction_entry.find(f"{dl_prefix}source/{dl_prefix}paragraphText")
    paragraph_text = paragraph_text.text if paragraph_text is not None else None

    reaction_smiles = reaction_entry.find(f"{dl_prefix}reactionSmiles")
    reaction_smiles = reaction_smiles.text if reaction_smiles is not None else None

    # ------------------------------------------------------------------------------------------------------------------
    # Step 2: Extract the reaction entry yield information.
    # ------------------------------------------------------------------------------------------------------------------
    reaction_product_yield_values = {}

    # First, iterate through the pre-parsed reaction products and analyze the related amount information.
    for reaction_product_ind, reaction_product in enumerate(reaction_entry.find(f"{cml_prefix}productList")):
        product_smiles = f"unknown_{reaction_product_ind}"

        for product_identifier in reaction_product.findall(f"{cml_prefix}identifier"):
            if "smiles" in product_identifier.attrib["dictRef"].lower():
                try:
                    product_mol = AllChem.MolFromSmiles(product_identifier.attrib["value"])
                    AllChem.SanitizeMol(product_mol)

                    product_smiles = AllChem.MolToSmiles(product_mol, canonical=True)

                except:
                    product_smiles = product_identifier.attrib["value"]

        reaction_product_yield_values[product_smiles] = set()

        for product_amount in reaction_product.findall(f"{cml_prefix}amount"):
            if f"{dl_prefix}propertyType" in product_amount.attrib:
                yield_type = product_amount.attrib[f"{dl_prefix}propertyType"].lower()
            else:
                yield_type = None

            if f"{dl_prefix}normalizedValue" in product_amount.attrib:
                parsed_normalized_value = parser.parse(product_amount.attrib[f"{dl_prefix}normalizedValue"])

                if len(parsed_normalized_value) > 0:
                    parsed_normalized_value = float(parsed_normalized_value[0].value)
                else:
                    parsed_normalized_value = None

            else:
                parsed_normalized_value = None

            parsed_text_values = parser.parse(product_amount.text)

            if len(parsed_text_values) == 0:
                reaction_product_yield_values[product_smiles].add((
                    yield_type,
                    None,
                    "percent" if "percent" in yield_type else None,
                    parsed_normalized_value,
                    "high"
                ))

            elif len(parsed_text_values) == 1:
                reaction_product_yield_values[product_smiles].add((
                    yield_type,
                    float(parsed_text_values[0].value),
                    "percent" if "percent" in yield_type else parsed_text_values[0].unit.name.lower(),
                    parsed_normalized_value,
                    "high"
                ))

            else:
                reaction_product_yield_values[product_smiles].add((
                    yield_type,
                    None,
                    "percent" if "percent" in yield_type else parsed_text_values[-1].unit.name.lower(),
                    parsed_normalized_value,
                    "high"
                ))

    # ------------------------------------------------------------------------------------------------------------------
    # Next, move all unassigned yield information to the nearest reaction product without yield information.

    unknown_key_info, known_key_info = [], []

    if any("unknown" in k for k in reaction_product_yield_values.keys()):
        for reaction_product_key_ind, reaction_product_key in enumerate(reaction_product_yield_values.keys()):
            if "unknown" in reaction_product_key:
                if len(reaction_product_yield_values[reaction_product_key]) > 0:
                    unknown_key_info.append(
                        (reaction_product_key_ind, reaction_product_yield_values[reaction_product_key])
                    )

            else:
                if len(reaction_product_yield_values[reaction_product_key]) == 0:
                    known_key_info.append(
                        (reaction_product_key_ind, reaction_product_key)
                    )

        if len(known_key_info) == 1 and len(unknown_key_info) == 1:
            if abs(known_key_info[0][0] - unknown_key_info[0][0]) == 1:
                reaction_product_yield_values[known_key_info[0][1]] = unknown_key_info[0][1]

    reaction_product_yield_values = {k: v for k, v in reaction_product_yield_values.items() if "unknown" not in k}

    # ------------------------------------------------------------------------------------------------------------------
    # Next, for all reaction products still without reaction yield information, continue the extraction by traversing
    # the pre-parsed, yield-related action phrases and relating them to such reaction products.

    search_further = False

    for reaction_product_value in reaction_product_yield_values.values():
        if len(reaction_product_value) == 0:
            search_further = True
            break

    if search_further:
        reaction_action_yield_values = []

        for reaction_action in reaction_entry.find(f"{dl_prefix}reactionActionList"):
            if "action" in reaction_action.attrib:
                if "yield" in reaction_action.attrib["action"].lower():
                    for parsed_action_yield in parser.parse(reaction_action.find(f"{dl_prefix}phraseText").text):
                        if parsed_action_yield.unit.entity.name.lower() == "mass":
                            reaction_action_yield_values.append((
                                "mass",
                                float(parsed_action_yield.value),
                                parsed_action_yield.unit.name.lower(),
                                None,
                                "medium"
                            ))

                        if parsed_action_yield.unit.name.lower() == "percent":
                            reaction_action_yield_values.append((
                                "percentyield",
                                float(parsed_action_yield.value),
                                "percent",
                                None,
                                "medium"
                            ))

        for reaction_action_yield_value in reversed(reaction_action_yield_values):
            for reaction_product_key in reversed(reaction_product_yield_values.keys()):
                if len(reaction_product_yield_values[reaction_product_key]) == 0:
                    reaction_product_yield_values[reaction_product_key].add(reaction_action_yield_value)

    # ------------------------------------------------------------------------------------------------------------------
    # Finally, for all reaction products still without reaction yield information, continue the extraction by parsing
    # the raw paragraph text and relating the extracted values to such reaction products.

    for reaction_product_value in reaction_product_yield_values.values():
        if len(reaction_product_value) == 0:
            search_further = True
            break

    if search_further:
        paragraph_text_yield_values = []

        for parsed_action_yield in parser.parse(paragraph_text[int(-len(paragraph_text)/3):]):
            if parsed_action_yield.unit.entity.name.lower() == "mass":
                paragraph_text_yield_values.append((
                    "mass",
                    float(parsed_action_yield.value),
                    parsed_action_yield.unit.name.lower(),
                    None,
                    "low"
                ))

            if parsed_action_yield.unit.name.lower() == "percent":
                paragraph_text_yield_values.append((
                    "percentyield",
                    float(parsed_action_yield.value),
                    "percent",
                    None,
                    "low"
                ))

        for paragraph_text_yield_value in reversed(paragraph_text_yield_values):
            for reaction_product_key in reversed(reaction_product_yield_values.keys()):
                if len(reaction_product_yield_values[reaction_product_key]) == 0:
                    reaction_product_yield_values[reaction_product_key].add(paragraph_text_yield_value)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 3: Extract the reaction entry time, temperature and purification method information.
    # ------------------------------------------------------------------------------------------------------------------
    reaction_time_values, reaction_temperature_values, reaction_purification_method_values = [], [], []

    for reaction_action in reaction_entry.find(f"{dl_prefix}reactionActionList"):
        reaction_purification_method_values.append((
            reaction_action.attrib["action"].lower(),
            reaction_action.find(f"{dl_prefix}phraseText").text
        ))

        for reaction_action_parameter in reaction_action.findall(f"{dl_prefix}parameter"):
            if reaction_action_parameter is not None:
                if "propertyType" in reaction_action_parameter.attrib:
                    if reaction_action_parameter.attrib["propertyType"].lower() == "time":
                        if "normalizedValue" in reaction_action_parameter.attrib:
                            parsed_time_value = parser.parse(reaction_action_parameter.attrib["normalizedValue"])

                            if len(parsed_time_value) > 0:
                                parsed_time_value = float(parsed_time_value[0].value)
                            else:
                                parsed_time_value = None

                        else:
                            parsed_time_value = None

                        reaction_time_values.append((
                            "time",
                            reaction_action.find(f"{dl_prefix}phraseText").text,
                            parsed_time_value,
                            "seconds"
                        ))

                    if reaction_action_parameter.attrib["propertyType"].lower() == "temperature":
                        if "normalizedValue" in reaction_action_parameter.attrib:
                            parsed_temperature_value = parser.parse(reaction_action_parameter.attrib["normalizedValue"])

                            if len(parsed_temperature_value) > 0:
                                parsed_temperature_value = float(parsed_temperature_value[0].value)
                            else:
                                parsed_temperature_value = None

                        else:
                            parsed_temperature_value = None

                        reaction_temperature_values.append((
                            "temperature",
                            reaction_action.find(f"{dl_prefix}phraseText").text,
                            parsed_temperature_value,
                            "degree celsius"
                        ))

    return document_id, heading_text, paragraph_num, paragraph_text, reaction_smiles, reaction_product_yield_values, \
           reaction_time_values, reaction_temperature_values, reaction_purification_method_values


def main():
    """ The main function of the script. """

    RDLogger.DisableLog("rdApp.*")

    args = get_parser()

    random.seed(args.random_seed)

    cml_files_folder_path = os.path.join(args.output_folder_path, "dataset_2_cml_files")

    if not os.path.exists(cml_files_folder_path):
        os.mkdir(cml_files_folder_path)

    all_reaction_entries = []

    for directory_name in os.listdir(args.input_folder_path):
        directory_reaction_entries = []

        for file_name in tqdm(iterable=os.listdir(os.path.join(args.input_folder_path, directory_name)),
                              total=len(os.listdir(os.path.join(args.input_folder_path, directory_name))),
                              ascii=True, ncols=120, desc=f"Processing the '{directory_name}' folder"):
            cml_root = ElementTree.parse(os.path.join(args.input_folder_path, directory_name, file_name)).getroot()

            with mp.Pool(args.num_cores) as process_pool:
                processing_results = [
                    (directory_name, ) + (file_name, ) + entry
                    for entry in process_pool.imap(process_xml_reaction_entry, cml_root)
                ]

                process_pool.close()
                process_pool.join()

            directory_reaction_entries.extend(processing_results)
            # break

        pd.DataFrame(
            data=directory_reaction_entries,
            columns=["publication_year", "cml_file", "document_id", "heading_text", "paragraph_num", "paragraph_text",
                     "reaction_smiles", "reaction_yield", "reaction_time", "reaction_temperature",
                     "reaction_purification_method"]
        ).to_pickle(os.path.join(cml_files_folder_path, f"{directory_name}_additional_information.pkl"))

        all_reaction_entries.extend(directory_reaction_entries)
        # break

    pd.DataFrame(
        data=all_reaction_entries,
        columns=["publication_year", "cml_file", "document_id", "heading_text", "paragraph_num", "paragraph_text",
                 "reaction_smiles", "reaction_yield", "reaction_time", "reaction_temperature",
                 "reaction_purification_method"]
    ).to_pickle(os.path.join(cml_files_folder_path, f"cml_files_additional_information.pkl"))


if __name__ == "__main__":
    main()

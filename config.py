import json

from argparse import ArgumentParser
from typing import List, NamedTuple, Optional


class TemplateExtractionConfig(NamedTuple):
    """
    Description:
        Class containing all of the neccessary configuration parameters for template extraction.
    """

    # Path to the input dataset file.    
    input_file_path: str
    # Extension of the input dataset file.
    input_file_extension: str
    # Row separator of the input dataset file.
    input_file_separator: str
    # Name of the column containing the ID's of the dataset entries.
    id_column: str
    # Name of the column containing the reaction SMILES strings of the dataset entries.
    rxn_smiles_column: str

    # Path to the folder where the output dataset file should be saved.
    output_folder_path: str
    # Name of the output dataset file.
    output_file_name: str
    # Extension of the output dataset file.
    output_file_extension: str
    # Row separator of the output dataset file.
    output_file_separator: str
    # Name of the new column that will be generated for the cleaned reaction SMILES strings.
    cleaned_smiles_column: str
    # Name of the new column that will be generated for the mapped reaction SMILES strings.
    mapped_smiles_column: str
    # Name of the new column that will be generated for the reaction template SMARTS strings
    rxn_template_smarts_column: str

    # Path to the '*.txt' file containing additional user-defined salts.
    salts_file_path: str

    # Timeout period in ms after which the atom mapping should be stopped for a single reaction SMILES.
    atom_mapping_timeout: int
    # Way to handle any previous mapping in the reaction SMILES string.
    handle_existing_mapping: str

    # Timeout period in ms after which the template extraction should be stopped for a single reaction SMILES.
    extract_template_timeout: int
    # The length of reaction SMILES which is not in danger of having a timeout, and can be processed faster. 
    extract_template_threshold: int
    # The number of occurences for a single reaction SMARTS based on which it is included in the final result.
    template_occurrence_threshold: int

    # Flag to signal if multiprocessing should be used or not.
    use_multiprocessing: bool


class DataPreProcessingConfig(NamedTuple):
    """
    Description:
        Class containing all of the neccessary configuration parameters for data pre-processing.
    """

    # Path to the input dataset file.    
    input_file_path: str
    # Extension of the input dataset file.
    input_file_extension: str
    # Row separator of the input dataset file.
    input_file_separator: str
    # Name of the column containing the ID's of the dataset entries.
    id_column: str
    # Name of the column containing the SMILES strings of the dataset entries.
    smiles_column: str

    # Path to the folder where the output dataset file should be saved.
    output_folder_path: str
    # Name of the output dataset file.
    output_file_name: str
    # Extension of the output dataset file.
    output_file_extension: str
    # Row separator of the output dataset file.
    output_file_separator: str
    # Name of the new column that will be generated for the canonical SMILES strings.
    processed_smiles_column: str
    # Name of the new column that will be generated for the SA_Score values.
    sa_score_column: str

    # Path to the '*.txt' file containing additional user-defined salts.
    salts_file_path: str
    # Path to the '*.sma' file containing the user-defined unwanted elements.
    unwanted_elements_file_path: str

    # Flag to signal if multiprocessing should be used or not.
    use_multiprocessing: bool


class Config(NamedTuple):
    """
    Description:
        Class containing the neccessary methods to load configuration files.
    """

    # Configuration parameters for the automatic template extraction task.
    template_extraction_config: TemplateExtractionConfig

    # Configuration parameters for the dataset pre-processing task.
    data_pre_processing_config: DataPreProcessingConfig

    @classmethod
    def load_configuration(cls, file_path: Optional[str] = None) -> "Config":
        """ Description: Load the configuration parameters from the specified configuration file. """

        # If the configuration file path is not specified, parse the arguments specified by the user input.
        if file_path is None:
            parser = ArgumentParser()
            parser.add_argument("-c", "--config", type=str, required=True, help="Path to the 'config.json' file.")
            args = parser.parse_args()
            file_path = args.config
        
        with open(file_path) as read_handle:
            settings = json.load(read_handle)

            if "template_extraction_config" not in settings or "data_pre_processing_config" not in settings:
                raise ValueError("Mandatory setting groups are missing from the configuration file.")
            
            return cls(template_extraction_config=TemplateExtractionConfig(**settings["template_extraction_config"]),
                       data_pre_processing_config=DataPreProcessingConfig(**settings["data_pre_processing_config"]))

import multiprocessing as mp
import pandas as pd

from tqdm import tqdm
from typing import List

from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt

from .utils.chem_utils import ConversionUtils, StructureUtils, SaScoreUtils
from .utils.data_utils import DataUtils


class DataPreProcessing:
    """ 
    Description:
        A class containing the implementation of the dataset pre-processing worklfow specified by RIKEN/Kyoto University.
    Configuration:
        Configuration parameters are defined and clarified in the 'config.json' and 'config.py' files, respectively.
    """
    
    def __init__(self, config_params):
        """"
        Description:
            Constructor to initialize the configuration parameters.
        """

        # Input data information.
        self.input_file_path = config_params.input_file_path
        self.input_file_extension = config_params.input_file_extension
        self.input_file_separator = config_params.input_file_separator
        self.id_column = config_params.id_column
        self.smiles_column = config_params.smiles_column
        
        # Output data information.
        self.output_folder_path = config_params.output_folder_path
        self.output_file_name = config_params.output_file_name
        self.output_file_extension = config_params.output_file_extension
        self.output_file_separator = config_params.output_file_separator
        self.processed_smiles_column = config_params.processed_smiles_column if config_params.processed_smiles_column != "" else "processed_canonical_smiles"
        self.sa_score_column = config_params.sa_score_column if config_params.sa_score_column != "" else "sa_score"
        
        # SMILES string cleaning information.
        self.salts_file_path = config_params.salts_file_path
        
        # In order to speed up the processing, the unwanted elements are converted into RDKit Mol objects only once in the constructor.
        if config_params.unwanted_elements_file_path == "":
            self.unwanted_elements = []
        else:
            self.unwanted_elements = [ConversionUtils.smarts_to_mol(uwe_smarts) for uwe_smarts in
                                      DataUtils.read_dataset(dataset_file_path=config_params.unwanted_elements_file_path,
                                                             dataset_file_extension= ".sma",
                                                             separator="\n", header=None)]
            self.unwanted_elements = [uw_element for uw_element in self.unwanted_elements if uw_element is not None]
        
        # General settings.
        self.use_multiprocessing = config_params.use_multiprocessing

        # Additional settings.
        pd.options.mode.chained_assignment = None
    
    def pre_process_entry(self, smiles: str, verbose=False) -> str:
        """ 
        Description:
            Pre-process a single molecule SMILES string by converting it to the canonical SMILES represantation, 
            removing the salts, checking for unwanted elements, and normalizing the molecular structure using RDKit.
        Input:
            smiles (str): A SMILES string of the molecule being processed.
        Output:
            (str): A cleaned Canonical SMILES string of the molecule.
        """

        # Convert the SMILES string into a Canonical SMILES representation.
        canonical_smiles = ConversionUtils.smiles_to_canonical_smiles(smiles, verbose=verbose)
        
        # Remove any salts that are present in the SMILES string. 
        if canonical_smiles is not None and "." in canonical_smiles:
            canonical_smiles = StructureUtils.remove_salts(canonical_smiles,
                                                           salts_file_path=self.salts_file_path,
                                                           apply_ad_hoc_stripper=True,
                                                           verbose=verbose)

        # Check if the SMILES string of the compound contains any of the unwanted elements like 
        # inappropriate substructures and non-druglike elements.
        if canonical_smiles is None or any(ConversionUtils.smiles_to_mol(smiles).HasSubstructMatch(uw_element)
                                           for uw_element in self.unwanted_elements):
            return None
        else:
            normalized_structure = StructureUtils.normalize_structure(canonical_smiles, verbose=verbose)

            # Check the consistency between the Canonical SMILES and normalized structure SMILES.
            if canonical_smiles != normalized_structure:
                return None
            else:
                return normalized_structure, SaScoreUtils.calculate_sa_score(normalized_structure)
    
    def _pre_process_dataset(self, compound_dataset: pd.DataFrame, id_column: str, smiles_column: str, 
                              processed_smiles_column: str, sa_score_column: str) -> pd.DataFrame:
        """ 
        Description:
            A worker function for the pre-processing of the raw input data. 
        Input:
            compound_dataset (pd.DataFrame): A Pandas DataFrame of the dataset that needs to be pre-processed.
            id_column (str): The name of the column containing the ID of the dataset entries.
            smiles_column (str): The name of the column containing the SMILES string of the dataset entries.
            processed_smiles_column (str): The name of the new column that will be generated containing the canonical SMILES string.
            sa_score_column (str): The name of the new column that will be generated containing the SA_Score.
        Output:
            (pd.DataFrame): A Pandas DataFrame of the pre-processed dataset.
        """
        
        # Remove duplicate rows according to the ID and SMILES columns and rows that are missing the SMILES string.
        if id_column != "":
            compound_dataset = compound_dataset.drop_duplicates(subset=[id_column])
        
        compound_dataset = compound_dataset.dropna(subset=[smiles_column])
        compound_dataset = compound_dataset[compound_dataset[smiles_column] != ""]

        # Generate a new column for Canonical SMILES strings, and remove duplicates and rows with missing values.
        if self.use_multiprocessing:
            process_pool = mp.Pool(mp.cpu_count()-2)
            
            compound_dataset[processed_smiles_column], compound_dataset[sa_score_column] = zip(*[(processed_entry[0], processed_entry[1]) 
                                                                                                 if processed_entry is not None else (None, None) for processed_entry in 
                                                                                                 tqdm(process_pool.imap(self.pre_process_entry,
                                                                                                                        compound_dataset[smiles_column].values),
                                                                                                                        total=len(compound_dataset[smiles_column].values),
                                                                                                                        ascii=True, desc="Processing entries (Multiple Cores)")])
                    
            process_pool.close()
            process_pool.join()

        else:
            compound_dataset[processed_smiles_column], compound_dataset[sa_score_column] = zip(*[(processed_entry[0], processed_entry[1]) 
                                                                                                 if processed_entry is not None else (None, None) for processed_entry in
                                                                                                 [self.pre_process_entry(smiles) for smiles in 
                                                                                                  tqdm(compound_dataset[smiles_column].values,
                                                                                                       ascii=True, desc="Processing entries (Single Core)")]])
        
        compound_dataset = compound_dataset.dropna(subset=[processed_smiles_column])
        compound_dataset = compound_dataset.drop_duplicates(subset=[processed_smiles_column])
        
        # Return the processed dataset sorted by the ID column.
        if id_column != "":
            return compound_dataset.sort_values(by=[id_column])
        else:
            return compound_dataset.sort_values(by=[sa_score_column])
    
    def pre_process_data(self, verbose=False):
        """ 
        Description:
            A user-friendly version fo the previous function for the pre-processing of the raw input data.
        """

        # Step 1: Read the input dataset that needs to be pre-processed.
        input_dataset = DataUtils.read_dataset(dataset_file_path=self.input_file_path,
                                               dataset_file_extension=self.input_file_extension,
                                               separator=self.input_file_separator)
        
        before_pre_processing = input_dataset.shape

        # Step 2: Pre-process the input dataset.
        input_dataset = self._pre_process_dataset(compound_dataset=input_dataset,
                                                  id_column=self.id_column,
                                                  smiles_column=self.smiles_column,
                                                  processed_smiles_column=self.processed_smiles_column,
                                                  sa_score_column=self.sa_score_column)
        
        if verbose:
            print("Dataset shape before pre-processing: {}".format(before_pre_processing))
            print("Dataset shape after pre-processing: {}".format(input_dataset.shape))
            print("Average SA_Score value: {}".format(round(input_dataset[self.sa_score_column].values.mean(), 2)))
            print(input_dataset.head(5))
        
        # Step 3: Save the pre-processed dataset.
        DataUtils.save_dataset(dataset_df=input_dataset,
                               output_folder_path=self.output_folder_path,
                               output_file_name=self.output_file_name,
                               output_file_extension=self.output_file_extension,
                               separator=self.output_file_separator,
                               verbose=verbose)

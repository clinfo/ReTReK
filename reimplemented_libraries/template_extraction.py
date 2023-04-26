import multiprocessing as mp
import pandas as pd

from collections import Counter
from functools import partial
from tqdm import tqdm
from typing import List

from multiprocessing.dummy import Pool as ThreadPool

from .utils.chem_utils import ConversionUtils, StructureUtils, ReactionMapper
from .utils.data_utils import DataUtils 


class TemplateExtraction:
    """ 
    Description:
        A class containing the implementation of the automatic extraction of reaction rule templates specified by RIKEN/Kyoto University.
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
        self.rxn_smiles_column = config_params.rxn_smiles_column

        # Output data information.
        self.output_folder_path = config_params.output_folder_path
        self.output_file_name = config_params.output_file_name
        self.output_file_extension = config_params.output_file_extension
        self.output_file_separator = config_params.output_file_separator
        self.cleaned_smiles_column = config_params.cleaned_smiles_column
        self.mapped_smiles_column = config_params.mapped_smiles_column
        self.rxn_template_smarts_column = config_params.rxn_template_smarts_column

        # Reaction SMILES cleaning information.
        self.salts_file_path = config_params.salts_file_path

        # Epam Indigo reaction SMILES mapping API information.
        self.atom_mapping_timeout = config_params.atom_mapping_timeout
        self.handle_existing_mapping = config_params.handle_existing_mapping

        # Reaction template extraction and filtering information.
        self.extract_template_timeout = int(config_params.extract_template_timeout / 1000)
        self.extract_template_threshold = config_params.extract_template_threshold
        self.template_occurrence_threshold = config_params.template_occurrence_threshold

        # General settings.
        self.use_multiprocessing = config_params.use_multiprocessing

        # Additional settings.
        pd.options.mode.chained_assignment = None
    
    def clean_entry(self, rxn_smiles: str, verbose=False) -> str:
        """
        Description:
            Clean a single reaction SMILES string entry using the approach described in the journal article.
        Input:
            rxn_smiles (str): A reaction SMILES string of a chemical reaction being processed.
        Output:
            (str): A cleaned reaction SMILES string of the chemical reaction.
        """

        # Extract neccessary reaction roles from the reaction SMILES string.
        reactant_smiles, _, product_smiles = ConversionUtils.rxn_smiles_to_rxn_roles(rxn_smiles, verbose=verbose)

        reactant_side = ".".join(reactant_smiles)
        product_side = ".".join(product_smiles)

        # Remove the user-defined salts.
        reactant_side = StructureUtils.remove_salts(reactant_side, salts_file_path=self.salts_file_path, verbose=verbose)
        product_side = StructureUtils.remove_salts(product_side, salts_file_path=self.salts_file_path, verbose=verbose)

        # Filter reactions that do not have any reactants or products at all.
        if reactant_side is None or reactant_side == "" or product_side is None or product_side == "":
            return None
        else:
            return ">>".join([reactant_side, product_side])
    
    def atom_map_entry(self, rxn_smiles: str, verbose=False) -> str:
        """
        Description:
            Map the reactive atoms for a single reaction SMILES entry using the Epam Indigo API.
        Input:
            rxn_smiles (str): A reaction SMILES string of a chemical reaction being processed.
        Output:
            (str): A mapped reaction SMILES string of the chemical reaction.
        """

        return ReactionMapper.atom_map_reaction(rxn_smiles,
                                                timeout_period=self.atom_mapping_timeout,
                                                existing_mapping=self.handle_existing_mapping,
                                                verbose=verbose)
    
    def extract_reaction_template_from_entry(self, mapped_rxn_smiles: str, verbose=False) -> str:
        """
        Description:
            Extract the reaction template from a mapped reaction SMILES string. 
        Input:
            mapped_rxn_smiles (str): A mapped reaction SMILES string of a chemical reaction being processed.
        Output:
            (str): A reaction template SMARTS string.
        """

        return ReactionMapper.extract_reaction_template(mapped_rxn_smiles, verbose=verbose)

    def _collect_result(self, func_result):
        """
        Description:
            Helper function to fetch the result of the asynchronously called function.
        """

        return func_result

    def _abortable_worker(self, func, *args, **kwargs):
        """
        Description:
            Helper function to implement a timeout for the Pool object.
        """
        
        timeout = kwargs.get("timeout", None)
        thread_pool_process = ThreadPool(1)
        func_result = thread_pool_process.apply_async(func, args=args)

        try:
            final_result = func_result.get(timeout)

            return final_result
        
        except mp.TimeoutError as ex:
            return None
    
    def _clean_reaction_dataset(self, reaction_dataset: pd.DataFrame, rxn_smiles_column: str, cleaned_smiles_column: str) -> pd.DataFrame:
        """
        Description:
            A worker function for the cleaning of reaction SMILES entries from a specified dataset.  
        Input:
            reaction_dataset (pd.DataFrame): A Pandas DataFrame of the reaction dataset that needs to be cleaned.
            rxn_smiles_column (str): The name of the column containing the reaction SMILES string of the dataset entries.
            cleaned_smiles_column (str): The name of the new column that will be generated containing the cleaned reaction SMILES string.
        Output:
            (pd.DataFrame): A Pandas DataFrame containing the cleaned reaction dataset.
        """

        # Remove duplicate rows according to the reaction SMILES strings.
        reaction_dataset = reaction_dataset.drop_duplicates(subset=[rxn_smiles_column])

        # Clean all of the reaction SMILES strings.
        if self.use_multiprocessing:
            process_pool = mp.Pool(mp.cpu_count())
            reaction_dataset[cleaned_smiles_column] = tqdm(process_pool.imap(self.clean_entry, reaction_dataset[rxn_smiles_column].values),
                                                           total=len(reaction_dataset[rxn_smiles_column].values), ascii=True,
                                                           desc="Cleaning reaction SMILES (Multiple Cores)")
            process_pool.close()
            process_pool.join()
        else:
            reaction_dataset[cleaned_smiles_column] = [self.clean_entry(rxn_smiles) for rxn_smiles in 
                                                       tqdm(reaction_dataset[rxn_smiles_column].values,
                                                            ascii=True, desc="Cleaning reaction SMILES (Single Core)")]
        
        # Drop all of the reactions with no reactant or product side and duplicate reaction SMILES strings.
        reaction_dataset = reaction_dataset.dropna(subset=[cleaned_smiles_column])
        reaction_dataset = reaction_dataset.drop_duplicates(subset=[cleaned_smiles_column])

        return reaction_dataset
    
    def _atom_map_reaction_dataset(self, reaction_dataset: pd.DataFrame, rxn_smiles_column: str, mapped_smiles_column: str) -> pd.DataFrame:
        """
        Description:
            A worker function for the atom mapping of the reaction SMILES entries using the Epam Indigo API.
        Input:
            reaction_dataset (pd.DataFrame): A Pandas DataFrame of the reaction dataset that needs to be cleaned.
            rxn_smiles_column (str): The name of the column containing the reaction SMILES string of the dataset entries.
            mapped_smiles_column (str): The name of the new column that will be generated containing the mapped reaction SMILES strings.
        Output:
            (pd.DataFrame): A Pandas DataFrame containing the mapped reaction dataset.
        """

        # Perform the atom mapping on all reaction SMILES strings.
        if self.use_multiprocessing:
            process_pool = mp.Pool(mp.cpu_count())
            reaction_dataset[mapped_smiles_column] = tqdm(process_pool.imap(self.atom_map_entry, reaction_dataset[rxn_smiles_column].values),
                                                          total=len(reaction_dataset[rxn_smiles_column].values), ascii=True,
                                                          desc="Mapping reaction SMILES (Multiple Cores)")
            process_pool.close()
            process_pool.join()
        else:
            reaction_dataset[mapped_smiles_column] = [self.atom_map_entry(rxn_smiles) for rxn_smiles in
                                                      tqdm(reaction_dataset[rxn_smiles_column].values, ascii=True,
                                                      desc="Mapping reaction SMILES (Single Core)")]
        
        return reaction_dataset
    
    def _extract_reaction_templates_from_dataset(self, reaction_dataset: pd.DataFrame, rxn_smiles_column: str, rxn_template_smarts_column: str) -> pd.DataFrame:
        """
        Description:
            A worker function for the extraction of reaction templates from the reaction SMILES entries using the RDChiral library.
        Input:
            reaction_dataset (pd.DataFrame): A Pandas DataFrame of the reaction dataset that needs to be cleaned.
            rxn_smiles_column (str): The name of the column containing the reaction SMILES string of the dataset entries.
            rxn_template_smarts_column (str): The name of the new column that will be generated containing the reaction template SMARTS strings.
        Output:
            (pd.DataFrame): A Pandas DataFrame containing the extracted reaction templates.
        """

        reaction_templates_async = []

        # Split the input Pandas DataFrame based on the length threshold for improved performance.
        short_subset = reaction_dataset.loc[reaction_dataset[rxn_smiles_column].str.len() <= self.extract_template_threshold]
        long_subset = reaction_dataset.loc[reaction_dataset[rxn_smiles_column].str.len() > self.extract_template_threshold]

        # Extract reaction templates from the short reaction SMILES using a fast approach.
        # This approach does not have the ability to cause a time-out and can get stuck on large reaction SMILES strings.
        if self.use_multiprocessing:
            process_pool = mp.Pool(mp.cpu_count())
            short_subset[rxn_template_smarts_column] = tqdm(process_pool.imap(self.extract_reaction_template_from_entry,
                                                                              short_subset[rxn_smiles_column].values),
                                                            total=len(short_subset[rxn_smiles_column].values), ascii=True,
                                                            desc="Extracting short reaction templates (Multiple Cores)")
            
            process_pool.close()
            process_pool.join()
        
        else:
            short_subset[rxn_template_smarts_column] = [self.extract_reaction_template_from_entry(rxn_smiles) for rxn_smiles in
                                                        tqdm(short_subset[rxn_smiles_column].values, ascii=True,
                                                             desc="Extracting short reaction templates (Single Core)")]
        
        # Extract reaction templates from the long reaction SMILES using a slow, more secure approach.
        # Note: Using the same approach for the single core processing because of the neccessary process timeout.
        if self.use_multiprocessing:
            process_pool = mp.Pool(mp.cpu_count())
            tqdm_desc = "Extracting long reaction templates (Multiple Cores)"
        else:
            process_pool = mp.Pool(1, maxtasksperchild=1)
            tqdm_desc = "Extracting long reaction templates (Single Core)"

        for rxn_smiles in tqdm(long_subset[rxn_smiles_column].values, ascii=True, desc=tqdm_desc):
            abortable_func = partial(self._abortable_worker,
                                     self.extract_reaction_template_from_entry,
                                     timeout=self.extract_template_timeout)
            
            reaction_templates_async.append(process_pool.apply_async(abortable_func, args=(rxn_smiles,),
                                                                     callback=self._collect_result).get())
        
        process_pool.close()
        process_pool.join()

        # Concatenate both results and return the final dataset.
        long_subset[rxn_template_smarts_column] = reaction_templates_async

        return pd.concat([short_subset, long_subset])
    
    def _filter_reaction_templates(self, all_reaction_templates: List[str], template_occurrence_threshold=50) -> List[str]:
        """
        Description:
            A worker function for the extraction of the most frequently occurring reaction templates.
        Input:
            all_reaction_templates (List[str]): A list of all reaction templates.
            template_occurrence_threshold (int): A threshold value for the occurrence of a single reaction template.
        Output:
            (List[str]): A list of all reaction templates which occurred at least the specified amount of times.
        """

        # Count the occurences of all reaction templates.
        all_reaction_templates = [ConversionUtils.rxn_smarts_to_rxn_smiles(reaction_template) for reaction_template in all_reaction_templates]
        template_counter = Counter([reaction_template for reaction_template in all_reaction_templates if reaction_template is not None])
        
        # Consider only the reaction templates whcih occurred at least 'template_occurrence_threshold' number of times.
        filtered_templates = [ConversionUtils.rxn_smiles_to_rxn_smarts(template_item[0]) for template_item in 
                              sorted(template_counter.items(), key=lambda item: item[1], reverse=True) if 
                              template_item[1] >= template_occurrence_threshold]
              
        return filtered_templates
    
    def _prepare_kgcn_templates(self, extended_dataset: pd.DataFrame, template_occurrence_threshold=50) -> pd.DataFrame:
        """
        Description:
            A worker function for the extraction of the most frequently occurring reaction templates for the kGCN model.
        Input:
            extended_dataset (pd.DataFrame): The extended dataset dataframe.
            template_occurrence_threshold (int): A threshold value for the occurrence of a single reaction template.
        Output:
            (List[str]): A list of all reaction templates which occurred at least the specified amount of times.
        """

        # Since kGCN considers only one product, take the first product as the target and standardize it.
        extended_dataset["products"] = [p.split(">>")[1].split(".")[0] for p in extended_dataset[self.mapped_smiles_column].values]
        extended_dataset["is_valid"] = [ConversionUtils.smiles_to_mol(p) is not None for p in extended_dataset["products"].values]
        extended_dataset = extended_dataset[extended_dataset["is_valid"]]

        # Convert the extracted products to SMARTS strings.
        extended_dataset["products"] = [ConversionUtils.mol_to_smarts(ConversionUtils.smiles_to_mol(p)) for p in extended_dataset["products"].values]
        
        # Consider only the reaction templates whcih occurred at least 'template_occurrence_threshold' number of times.
        extended_dataset = extended_dataset[extended_dataset.groupby(
            self.rxn_template_smarts_column
        )[self.rxn_template_smarts_column].transform("size") >= template_occurrence_threshold]

        # Prepare the dataset columns.
        extended_dataset["reaction_core"] = extended_dataset[self.rxn_template_smarts_column]
        extended_dataset["max_publication_year"] = extended_dataset["Year"]
        extended_dataset["product"] = extended_dataset["products"]
        extended_dataset = extended_dataset[["product", "reaction_core", "max_publication_year"]]

        # Consider only templates which have one product.
        extended_dataset["n_products"] = [len(p.split(">>")[1].split(".")) for p in extended_dataset["reaction_core"]]
        extended_dataset = extended_dataset[extended_dataset["n_products"]==1]

        extended_dataset = extended_dataset.drop_duplicates(subset=["product", "reaction_core"], keep="last")

        return extended_dataset
    
    def extract_reaction_templates(self, save_extended_dataset=True, verbose=False):
        """ 
        Description:
            An user-friendly version of the automatic template extraction function.
        """

        # Step 1: Read the input dataset that needs to be pre-processed.
        reaction_dataset = DataUtils.read_dataset(dataset_file_path=self.input_file_path,
                                                  dataset_file_extension=self.input_file_extension,
                                                  separator=self.input_file_separator)
        
        original_dataset_shape = reaction_dataset.shape

        # Step 2: Clean the reaction SMILES in the dataset.
        reaction_dataset = self._clean_reaction_dataset(reaction_dataset=reaction_dataset,
                                                        rxn_smiles_column=self.rxn_smiles_column,
                                                        cleaned_smiles_column=self.cleaned_smiles_column)
        
        shape_after_cleaning = reaction_dataset.shape

        # Step 3: Perform atom mapping using the Epam Indigo API on the reaction SMILES in the dataset.
        reaction_dataset = self._atom_map_reaction_dataset(reaction_dataset=reaction_dataset,
                                                           rxn_smiles_column=self.cleaned_smiles_column,
                                                           mapped_smiles_column=self.mapped_smiles_column)
        
        successfully_mapped_entries = reaction_dataset.dropna(subset=[self.mapped_smiles_column]).shape
        
        # Step 4: Extract the reaction templates from the mapped reaction SMILES in the dataset.
        reaction_dataset = self._extract_reaction_templates_from_dataset(reaction_dataset=reaction_dataset,
                                                                         rxn_smiles_column=self.mapped_smiles_column,
                                                                         rxn_template_smarts_column=self.rxn_template_smarts_column)
        
        successfully_extracted_templates = reaction_dataset.dropna(subset=[self.rxn_template_smarts_column]).shape

        # Step 5: Filter the extracted reaction templates keeping only the ones which occurred the specified number of times.
        final_templates = self._filter_reaction_templates(reaction_dataset[self.rxn_template_smarts_column].values.tolist(),
                                                          template_occurrence_threshold=self.template_occurrence_threshold)
        
        # Step 6: Save the extracted reaction templates in format ready for kGCN processing.
        kgcn_templates_dataset = self._prepare_kgcn_templates(reaction_dataset, template_occurrence_threshold=self.template_occurrence_threshold)
        
        # Step 7: Print a short summary of the extraction process, if it is indicated.
        if verbose:
            print("\nOriginal Dataset Shape: {}".format(original_dataset_shape))
            print("Dataset Shape After Cleaning: {}".format(shape_after_cleaning))
            print("Successfully Mapped Reactions (Epam Indigo): {}".format(successfully_mapped_entries))
            print("Successfully Extracted Templates (RDChiral): {}".format(successfully_extracted_templates))
            print("Number of Filtered Templates (x{}): {}".format(self.template_occurrence_threshold, len(final_templates)))
            print("Number of Valid kGCN Templates (x{}): {}".format(self.template_occurrence_threshold, len(kgcn_templates_dataset.index)))
        
        # Step 8: Save the extended dataset and the in the specified output folder.
        if save_extended_dataset:
            if verbose:
                print("Saving the extended reaction dataset as "
                      "'{}extended_dataset.csv'...".format(self.output_folder_path), end="", flush=True)
            
            DataUtils.save_dataset(reaction_dataset,
                                   output_folder_path=self.output_folder_path,
                                   output_file_name="extended_dataset",
                                   output_file_extension=".csv",
                                   separator=",")
            if verbose:
                print("done.")
        
        # Step 9: Save the filtered templates .sma in the specified output folder.
        if verbose:
            print("Saving the filtered reaction templates as '{}_x{}{}'...".format(self.output_folder_path +
                                                                                   self.output_file_name,
                                                                                   self.template_occurrence_threshold,
                                                                                   self.output_file_extension),
                                                                                   end="", flush=True)
        
        DataUtils.save_dataset(pd.DataFrame({"smarts": final_templates}),
                               output_folder_path=self.output_folder_path,
                               output_file_name=self.output_file_name + "_x{}".format(self.template_occurrence_threshold),
                               output_file_extension=self.output_file_extension,
                               separator="\n",
                               header=None)
        
        if verbose:
            print("done.")

        # Step 10: Save the filtered templates ready for kGCN processing in the specified output folder.
        if verbose:
            print("Saving the kGCN input file as 'kgcn_x{}.csv'...".format(self.template_occurrence_threshold), end="", flush=True)
        
        DataUtils.save_dataset(kgcn_templates_dataset,
                               output_folder_path=self.output_folder_path,
                               output_file_name="kgcn_x{}".format(self.template_occurrence_threshold),
                               output_file_extension=".csv",
                               separator=",")
        if verbose:
            print("done.")
                
        return final_templates

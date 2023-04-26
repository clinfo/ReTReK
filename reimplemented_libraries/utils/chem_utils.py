import re
import numpy as np

from indigo import *
from typing import List, Tuple, Union
from molvs import Standardizer
from rdkit.Chem import AllChem, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.main import rdchiralRun, rdchiralRunText
from rdchiral.template_extractor import extract_from_reaction

from .rdkit_sa_score import sascorer


class ConversionUtils:
    """
    Description:
        Class containing a group of methods for handling the conversion of chemical formats using RDKit.
    """

    @staticmethod
    def smiles_to_mol(smiles: str, verbose=False) -> AllChem.Mol:
        """
        Description:
            Convert a SMILES string to a RDKit Mol object.
            Returns None if either the conversion or the sanitization of the SMILES string fail.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (AllChem.Mol): An RDKit Mol object representing the given SMILES string.
        """

        mol = None

        # Try to convert the SMILES string into a RDKit Mol object and sanitize it.  
        try:
            mol = AllChem.MolFromSmiles(smiles)
            AllChem.SanitizeMol(mol)

            return mol
        
        # If an exception occurs for any reason, print the error message if indicated, and return None.
        except Exception as exception:
            if verbose:
                if mol is None:
                    print("Exception occured during the conversion process of ", end="")
                else:
                    print("Exception occured during the sanitization of ", end="")
                        
                print("'{}'. Detailed exception message:\n{}".format(smiles, exception))
            
            return None
    
    @staticmethod
    def smarts_to_mol(smarts: str, verbose=False) -> AllChem.Mol:
        """
        Description:
            Convert a SMARTS string to a RDKit Mol object.
            Returns None if either the conversion or the sanitization of the SMARTS string fail.
        Input:
            smarts (str): A SMARTS string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (AllChem.Mol): An RDKit Mol object representing the given SMARTS string.
        """

        smarts_mol = None

        # Try to convert the SMARTS string into a RDKit Mol object and sanitize it.  
        try:
            smarts_mol = AllChem.MolFromSmarts(smarts)
            AllChem.SanitizeMol(smarts_mol)

            return smarts_mol
        
        # If an exception occurs for any reason, print the error message if indicated, and return None.
        except Exception as exception:
            if verbose:
                if smarts_mol is None:
                    print("Exception occured during the conversion process of ", end="")
                else:
                    print("Exception occured during the sanitization of ", end="")
                        
                print("'{}'. Detailed exception message:\n{}".format(smarts, exception))
            
            return None
    
    @staticmethod
    def smiles_to_canonical_smiles(smiles: str, verbose=False) -> str:
        """
        Description:
            Convert a SMILES string to a Canonical SMILES string.
            Returns None if either the conversion to the Canonical SMILES string fails.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical SMILES string representing the given chemical structure.
        """

        try:
            return AllChem.MolToSmiles(ConversionUtils.smiles_to_mol(smiles), canonical=True)
        
        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print("Exception occured during the conversion of '{}' to Canonical SMILES. Detailed message: {}".format(smiles, exception))
            
            return None
            
    @staticmethod
    def mol_to_canonical_smiles(mol: AllChem.Mol, verbose=False) -> str:
        """
        Description:
            Convert a RDKit Mol object to a Canonical SMILES string.
            Returns None if either the conversion to the Canonical SMILES string fails.
        Input:
            mol (AllChem.Mol): An RDKit Mol object representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical SMILES string representing the given chemical structure.
        """

        try:
            return AllChem.MolToSmiles(mol, canonical=True)
        
        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print("Exception occured during the conversion of the RDKit Mol object to Canonical SMILES. Detailed message: {}".format(exception))
            
            return None
    
    @staticmethod
    def mol_to_smarts(mol: AllChem.Mol, verbose=False) -> str:
        """
        Description:
            Convert a RDKit Mol object to a SMARTS string.
            Returns None if either the conversion to the SMARTS string fails.
        Input:
            mol (AllChem.Mol): An RDKit Mol object representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A SMARTS string representing the given chemical structure.
        """

        try:
            return AllChem.MolToSmarts(mol)
        
        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print("Exception occured during the conversion of the RDKit Mol object to SMARTS. Detailed message: {}".format(exception))
            
            return None
    
    @staticmethod
    def rxn_smarts_to_rxn_smiles(rxn_smarts: str, verbose=False) -> str:
        """
        Description:
            Convert a reaction SMARTS string to a Canonical reaction SMILES string. 
        Input:
            rxn_smarts (str): A reaction SMARTS string representing a chemical reaction.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical reaction SMILES string.
        """

        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            AllChem.SanitizeRxn(rxn)

            return AllChem.ReactionToSmiles(rxn, canonical=True)
        
        # If an exception occurs for any reason, print the message if indicated, and return None for each of the reaction roles.
        except Exception as exception:
            if verbose:
                print("Exception occured during the conversion for reaction SMARTS '{}'. Detailed message: {}".format(rxn_smarts, exception))
            
            return None
    
    @staticmethod
    def rxn_smiles_to_rxn_smarts(rxn_smiles: str, verbose=False) -> str:
        """
        Description:
            Convert a reaction SMILES string to a reaction SMARTS string. 
        Input:
            rxn_smarts (str): A reaction SMILES string representing a chemical reaction.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A reaction SMARTS string.
        """

        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smiles)
            AllChem.SanitizeRxn(rxn)

            return AllChem.ReactionToSmarts(rxn)
        
        # If an exception occurs for any reason, print the message if indicated, and return None for each of the reaction roles.
        except Exception as exception:
            if verbose:
                print("Exception occured during the conversion for reaction SMILES '{}'. Detailed message: {}".format(rxn_smarts, exception))
            
            return None
    
    @staticmethod
    def rxn_smiles_to_rxn_roles(rxn_smiles: str, verbose=False) -> Tuple[List["str"]]:
        """
        Description:
            Parse the reaction roles strings from the reaction SMILES string. 
        Input:
            rxn_smiles (str): A SMILES string representing a chemical reaction.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (Tuple[List["str"]]): A 3-tuple containg the Canonical SMILES strings for the reactants, agents and products, respectively.
        """

        try:
            # Split the reaction SMILES string by the '>' symbol to obtain the reactants and products.
            # Sometimes, extended SMILES can have additional information at the end separated by a whitespace.
            reactant_smiles = [r_smi for r_smi in rxn_smiles.split(">")[0].split(".") if r_smi != ""]
            agent_smiles = [a_smi for a_smi in rxn_smiles.split(">")[1].split(".") if a_smi != ""]
            product_smiles = [p_smi for p_smi in rxn_smiles.split(">")[2].split(" ")[0].split(".") if p_smi != ""]

            return reactant_smiles, agent_smiles, product_smiles
            
        # If an exception occurs for any reason, print the message if indicated, and return None for each of the reaction roles.
        except Exception as exception:
            if verbose:
                print("Exception occured during the parsing of the reaction roles for '{}'. Detailed message: {}".format(rxn_smiles, exception))
            
            return None, None, None


class ReactionMapper:
    """
    Description:
        Class containing a multiprocessing-friendly wrapper for the Epam Indigo chemical reaction mapping API.
    """
    
    @staticmethod
    def atom_map_reaction(rxn_smiles: str, timeout_period: int, existing_mapping="discard", verbose=False) -> str:
        """ 
        Description: 
            Atom map a reaction SMILES string using the Epam Indigo reaction atom mapper API. 
            Any existing mapping will be handled according to the value of the parameter 'existing_mapping'.
            Because it can be a time consuming process, a timeout occurs after 'timeout_period' ms. 
        Input:
            rxn_smiles (str): A reaction SMILES string representing a chemical reaction which is going to be mapped.
            timeout_period (int): A timeout which occurs after the set number of ms. 
            existing_mapping (str): Method to handle any existing mapping: 'discard', 'keep', 'alter' or 'clear'.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): The mapped reaction SMILES string.
        """
        
        try:
            # Instantiate the Indigo class object and set the timeout period.
            indigo_mapper = Indigo()
            indigo_mapper.setOption("aam-timeout", timeout_period)
            
            # Return the atom mapping of the reaction SMILES string.
            rxn = indigo_mapper.loadReaction(rxn_smiles)
            rxn.automap(existing_mapping)

            return rxn.smiles()
        
        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print("Exception occured during atom mapping of the reaction SMILES. Detailed message: {}".format(exception))
            
            return None

    @staticmethod
    def extract_reaction_template(rxn_smiles: str, verbose=False) -> str:
        """
        Description: 
            Extract a reaction template from a SMILES string using the RDChiral library.  
            This function relies on the GetSubstructMatches function from RDKit and if the
            reaction contains many large molecules, the process can take a lot of time.
        Input:
            rxn_smiles (str): A reaction SMILES string from which the template is going to be extracted.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): The extracted reaction template in the form of a SMARTS string.
        """
        
        try:
            # Parse the reaction roles from the reaction SMILES.
            reactant_smiles, _, product_smiles = ConversionUtils.rxn_smiles_to_rxn_roles(rxn_smiles)
            
            reactant_side = ".".join(reactant_smiles)
            product_side = ".".join(product_smiles)

            if not verbose:
                # Prevent function from printing junk.
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
                
                # Extract the templates from the reaction SMILES using RDChiral.
                reaction_template = extract_from_reaction({"reactants": reactant_side, "products": product_side, "_id": "0"})

                sys.stdout = old_stdout
            
            else:
                # Extract the templates from the reaction SMILES using RDChiral.
                reaction_template = extract_from_reaction({"reactants": reactant_side, "products": product_side, "_id": "0"})

            # Return the reaction SMARTS result if the processing finished correctly. 
            if reaction_template is not None and "reaction_smarts" in reaction_template.keys():
                # Because RDChiral returns the switched template, switch the order of the reactants and products.
                reactant_side, _, product_side = reaction_template["reaction_smarts"].split(">")
                final_reaction_template = ">>".join([product_side, reactant_side])

                return final_reaction_template
            else:
                return None
            
        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print("Exception occured during the reaction rule template extraction from the reaction SMILES. Detailed message: {}".format(exception))
            
            return None


class PyReactor:
    """ 
    Description:
        A class containing a collection of static methods for the application of reaction templates in 
        the forward and backward direction.
    """

    @staticmethod
    def __remove_duplicate_molecules(mol_combinations: List[Tuple[AllChem.Mol]]) -> List[Tuple[str]]:
        """ 
        Description:
            Remove the duplicate molecule combination tuples from a given list.
        Input:
            mol_combinations (List[Tuple[AllChem.Mol]]): The list of molecule combination tuples.
        Output:
            (List[Tuple[str]]): The same collection of molecule combination tuples without duplicates,
                                and in canonical SMILES tuple format.
        """

        correct_mol_combinations = []

        for mol_combination in mol_combinations:
            mol_combination_tuple = tuple([ConversionUtils.mol_to_canonical_smiles(mol) for mol in mol_combination])

            if None in mol_combination_tuple:
                continue
            else:
                correct_mol_combinations.append(mol_combination_tuple)
        
        return list(set(correct_mol_combinations))
    
    @staticmethod
    def __remove_duplicate_molecules_rdchiral(mol_combinations: List[Tuple[str]]) -> List[Tuple[str]]:
        """ 
        Description:
            Remove the duplicate molecule combination tuples from a given list.
        Input:
            mol_combinations (List[Tuple[AllChem.Mol]]): The list of molecule combination tuples.
        Output:
            (List[Tuple[str]]): The same collection of molecule combination tuples without duplicates,
                                and in canonical SMILES tuple format.
        """

        correct_mol_combinations = []

        for mol_combination in mol_combinations:
            mol_combination_tuple = tuple([ConversionUtils.smiles_to_canonical_smiles(smiles) for smiles in mol_combination])

            if None in mol_combination_tuple:
                continue
            else:
                correct_mol_combinations.append(mol_combination_tuple)
        
        return list(set(correct_mol_combinations))
    
    @staticmethod
    def __check_suggested_mol_validity(mol_combinations: List[Tuple[str]]) -> List[Tuple[AllChem.Mol]]:
        """ 
        Description:
            Check the chemical validity of the molecules in a list of combination tuples.
        Input:
            mol_combinations (List[Tuple[str]]): The list of molecule SMILES string combination tuples.
        Output:
            (List[Tuple[AllChem.Mol]]): The same collection of molecule combination tuples containing only
                                        valid molecules in RDKit Mol object tuple format.
        """
        
        correct_mol_combinations = []

        for mol_combination in mol_combinations:
            mol_combination_tuple = tuple([ConversionUtils.smiles_to_mol(mol_smiles) 
                                           for mol_smiles in mol_combination])

            if None in mol_combination_tuple:
                continue
            else:
                correct_mol_combinations.append(mol_combination_tuple)
            
        return correct_mol_combinations
    
    @staticmethod
    def forward_apply(reactant_mols: Union[str, AllChem.Mol, Tuple[str], List[str], Tuple[AllChem.Mol], List[AllChem.Mol]],
                      rxn_template: str, return_type="mol") -> List[Tuple]:
        """ 
        Description:
            Apply a specified reaction rule template on a single or a group of reactant molecules.
        Input:
            reactant_mols (List[str], Tuple[str], List[AllChem.Mol], Tuple[AllChem.Mol]): 
                The collection of reactant SMILES strings or RDKit Mol objects.
            rxn_template (str): The reaction template based on which the reactants are determined.
            return_type (str): The format in which the results are returned. Can be 'str' for SMILES strings,
                               or 'mol' for RDKit Mol objects.
        Output:
            (List[Tuple]): The list of potential product combination tuples generated by PyReactor.
        """

        if rxn_template == "":
            raise Exception("The input reaction template is empty.")

        if isinstance(reactant_mols, str) or isinstance(reactant_mols, AllChem.Mol):
            if isinstance(reactant_mols, str):
                reactant_mols = [ConversionUtils.smiles_to_mol(reactant_mols)]
            else:
                reactant_mols = [reactant_mols]
        else:
            if len(reactant_mols) == 0:
                raise Exception("The input collection of reactants is empty.")
            
            if isinstance(reactant_mols[0], str):
                reactant_mols = [ConversionUtils.smiles_to_mol(mol) for mol in reactant_mols]
        
        if None in reactant_mols:
            raise Exception("Not all reactants could be successfully converted to RDKit Mol objects. "
                            "Please check the validity of the given reactant SMILES strings.")
        else:
            if len(rxn_template.split(">")[0].split(".")) != len(reactant_mols):
                raise Exception("The number of given reactants does not match the number of reactants in the template.")
        
        # Try to create and sanitize the RDKit ChemicalReaction object from the given reaction template.
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_template)
            AllChem.SanitizeRxn(rxn)

            # Remove any potential duplicate suggestions.
            products_suggestions_str = PyReactor.__remove_duplicate_molecules(rxn.RunReactants(reactant_mols))
            
            if return_type == "str":
                return products_suggestions_str
            else:
                # Convert the SMILES strings into RDKit Mol objects and check their chemical validity.
                return PyReactor.__check_suggested_mol_validity(products_suggestions_str)
            
        except Exception as exception:
            raise Exception("Exception occured during the construction of the ChemicalReaction object from "
                            "the reaction SMARTS '{}'. Please make sure that the reaction template is correct. "
                            "Detailed exception message: \n{}".format(rxn_template, exception))
        
    @staticmethod
    def reverse_apply_rdchiral(product_mols: Union[str, AllChem.Mol,  Tuple[str], List[str], Tuple[AllChem.Mol], List[AllChem.Mol]],
                               rxn_template: str, max_products=10, return_type="mol") -> List[Tuple]:
        """
        Description:
            Apply a specified reaction rule template backwards on a single or group of product molecules using the RDChiral library.
        Input:
            product_mols (List[str], Tuple[str], List[AllChem.Mol], Tuple[AllChem.Mol]): 
                The collection of product SMILES strings or RDKit Mol objects.
            rxn_template (str): The reaction template based on which the products are determined.
            return_type (str): The format in which the results are returned. Can be 'str' for SMILES strings,
                               or 'mol' for RDKit Mol objects.
        Output:
            (List[Tuple]): The list of potential reactant combination tuples generated by PyReactor.
        """

        if rxn_template == "":
            raise Exception("The input reaction template is empty.")

        if max_products <= 0:
            raise Exception("The number of generated products needs to be a positive number larger than 0.")
        
        if isinstance(product_mols, str) or isinstance(product_mols, AllChem.Mol):
            if isinstance(product_mols, str):
                product_mols = [product_mols]
            else:
                product_mols = [ConversionUtils.mol_to_canonical_smiles(product_mols)]
        else:
            if len(product_mols) == 0:
                raise Exception("The input collection of products is empty.")

            if isinstance(product_mols[0], AllChem.Mol):
                product_mols = [ConversionUtils.mol_to_canonical_smiles(mol) for mol in product_mols]
        
        if None in product_mols:
            raise Exception("Not all products could be successfully converted to RDKit Mol objects. "
                            "Please check the validity of the given product SMILES strings.")
        else:
            if len(rxn_template.split(">")[2].split(" ")[0].split(".")) != len(product_mols):
                raise Exception("The number of given products does not match the number of products in the template.")

        try:
            # Split the reaction rule template SMARTS string into reactants and products substrings.
            reactants_substr, _, products_substr = rxn_template.split(">")

            # Generate a reverse reaction SMARTS string.
            rxn_template = ">>".join([products_substr, reactants_substr])
            
            # Run the textual RDChiral suggestion generator.
            reactants_suggestions = rdchiralRunText(rxn_template, ".".join(product_mols))
            reactants_suggestions = [tuple(rs.split(".")) for rs in reactants_suggestions]
            
            # Remove any potential duplicate suggestions.
            reactants_suggestions_str = PyReactor.__remove_duplicate_molecules_rdchiral(reactants_suggestions)

            # Sort results to allow reproducibility.
            reactants_suggestions_str.sort()

            if return_type == "str":
                return reactants_suggestions_str[0:max_products]
            else:
                # Convert the SMILES strings into RDKit Mol objects and check their chemical validity.
                return PyReactor.__check_suggested_mol_validity(reactants_suggestions_str)[0:max_products]
        
        except Exception as exception:
            raise Exception("Exception occured during the construction of the RDChiral ChemicalReaction object from "
                            "the reaction SMARTS '{}'. Please make sure that the reaction template is correct. "
                            "Detailed exception message: \n{}".format(rxn_template, exception))
    
    @staticmethod
    def reverse_apply(product_mols: Union[str, AllChem.Mol, Tuple[str], List[str], Tuple[AllChem.Mol], List[AllChem.Mol]],
                      rxn_template: str, max_products=10, return_type="mol") -> List[Tuple]:
        """
        Description:
            Apply a specified reaction rule template backwards on a single or group of product molecules.
        Input:
            product_mols (List[str], Tuple[str], List[AllChem.Mol], Tuple[AllChem.Mol]): 
                The collection of product SMILES strings or RDKit Mol objects.
            rxn_template (str): The reaction template based on which the products are determined.
            return_type (str): The format in which the results are returned. Can be 'str' for SMILES strings,
                               or 'mol' for RDKit Mol objects.
        Output:
            (List[Tuple]): The list of potential reactant combination tuples generated by PyReactor.
        """

        if rxn_template == "":
            raise Exception("The input reaction template is empty.")

        if max_products <= 0:
            raise Exception("The number of generated products needs to be a positive number larger than 0.")
        
        if isinstance(product_mols, str) or isinstance(product_mols, AllChem.Mol):
            if isinstance(product_mols, str):
                product_mols = [ConversionUtils.smiles_to_mol(product_mols)]
            else:
                product_mols = [product_mols]
        else:
            if len(product_mols) == 0:
                raise Exception("The input collection of products is empty.")
            
            if isinstance(product_mols[0], str):
                product_mols = [ConversionUtils.smiles_to_mol(mol) for mol in product_mols]
        
        if None in product_mols:
            raise Exception("Not all products could be successfully converted to RDKit Mol objects. "
                            "Please check the validity of the given product SMILES strings.")
        else:
            if len(rxn_template.split(">")[2].split(" ")[0].split(".")) != len(product_mols):
                raise Exception("The number of given products does not match the number of products in the template.")
        
        try:
            # Split the reaction rule template SMARTS string into reactants and products substrings.
            reactants_substr, _, products_substr = rxn_template.split(">")
            
            # Generate a reverse reaction rule template SMARTS string.
            rxn_template = ">>".join([products_substr, reactants_substr])
            
            # Try to create and sanitize the RDKit ChemicalReaction object.
            rxn = AllChem.ReactionFromSmarts(rxn_template)
            AllChem.SanitizeRxn(rxn)
            
            # Remove any potential duplicate suggestions.
            reactants_suggestions_str = PyReactor.__remove_duplicate_molecules(rxn.RunReactants(product_mols, 
                                                                               maxProducts=max_products))
            
            # Sort results to allow reproducibility.
            reactants_suggestions_str.sort()
            
            if return_type == "str":
                return reactants_suggestions_str
            else:
                # Convert the SMILES strings into RDKit Mol objects and check their chemical validity.
                return PyReactor.__check_suggested_mol_validity(reactants_suggestions_str)
        
        except Exception as exception:
            raise Exception("Exception occured during the construction of the ChemicalReaction object from "
                            "the reaction SMARTS '{}'. Please make sure that the reaction template is correct. "
                            "Detailed exception message: \n{}".format(rxn_template, exception))


class StructureUtils:
    """
    Description:
        Class containing a group of methods for handling the correctness of molecular structures.
    """

    @staticmethod
    def remove_salts(smiles: str, salts_file_path: str, apply_ad_hoc_stripper=False, verbose=False) -> str:
        """
        Description:
            Remove salts from a SMILES string using the RDKit salt stripper.
            Returns None if the RDKit salt removal process fails.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            salt_list_file_path (str): A path string to a user-defined list of salt SMILES in .txt format.
            apply_ad_hoc_stripper (bool): A bool value indicating if the ad-hoc salt stripper will be applied.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical SMILES string representing the given chemical structure without salts.
        """

        try:
            # Apply the RDKit salt stripper to remove the defined salt molecules.
            salt_remover = SaltRemover.SaltRemover(defnFilename=salts_file_path)
            no_salt_smiles = ConversionUtils.mol_to_canonical_smiles(salt_remover.StripMol(ConversionUtils.smiles_to_mol(smiles)))
            
            # If there are some salts left behind, apply the 'ad hoc' salt stripper based on the symbol '.'.
            # NOTE: This is risky and it should only be applied if the SMILES string is one molecule, not on reaction SMILES.
            if apply_ad_hoc_stripper and "." in no_salt_smiles:
                no_salt_smiles = ConversionUtils.smiles_to_canonical_smiles(sorted(no_salt_smiles.split("."), key=len, reverse=True)[0])

            # If nothing is left behind because all of the molecule parts are defined as salts, return None.
            if no_salt_smiles == "":
                return None
            else:
                return no_salt_smiles
        
        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print("Exception occured during stripping of the salts from '{}'. Detailed exception message:\n{}".format(smiles, exception))
            
            return None
    
    @staticmethod
    def normalize_structure(smiles: str, verbose=False) -> str:
        """ 
        Description:
            Use RDKit to normalize the specified molecule and return it as canonical SMILES.
            Returns None if the RDKit normalization process fails.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical SMILES string representing the normalized chemical structure.
        """

        try: 
            mol = rdMolStandardize.Normalize(ConversionUtils.smiles_to_mol(smiles))

            return ConversionUtils.mol_to_canonical_smiles(mol)
        
        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print("Exception occurred during the normalization of '{}'. Detailed exception message:\n{}".format(smiles, exception))
            
            return None

    @staticmethod
    def molvs_standardize(smiles: str, verbose=False):
        """ 
        Description:
            Use MolVS and RDKit to standardize the specified molecule and return it as canonical SMILES.
            Returns None if the standardization process fails.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical SMILES string representing the standardized chemical structure.
        """

        try: 
            standardizer = Standardizer()
            standardized_mol = standardizer.standardize(ConversionUtils.smiles_to_mol(smiles))
            
            return ConversionUtils.mol_to_canonical_smiles(standardized_mol)
        
        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print("Exception occurred during the standardization of '{}'. Detailed exception message:\n{}".format(smiles, exception))
            
            return None

class SaScoreUtils:
    """
    Description:
        Class containing a group of methods for handling the SA_Score calculations for molecular structures.
    """
    
    @staticmethod
    def calculate_sa_score(smiles: str, verbose=False) -> float:
        """ 
        Description:
            Calculates the SA_Score value for a given SMILES string.
            Returns None if the calculation fails for any reason.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (float): A floating point value representing the SA_Score value for the input SMILES string.
        """

        # Try converting the current SMILES string into a RDKit Mol object and calculate the SA_Score value.
        try:
            mol = ConversionUtils.smiles_to_mol(smiles)
            AllChem.SanitizeMol(mol)

            return sascorer.calculateScore(mol)
        
        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print("Exception occurred during the SA_Score calculation for '{}'. Detailed exception message:\n{}".format(smiles, exception))
            
            return None

from argparse import ArgumentParser
from config import Config

from reimplemented_libraries.template_extraction import TemplateExtraction


def parse_user_args():
    """
    Description:
        Parse the arguments specified by the user during input.
    """

    parser = ArgumentParser("Template extraction")

    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("-t", "--task", default="all", type=str, choices=["clean", "map", "ext_tmplt", "all"], help="Task to perform.")
    parser.add_argument("-rsmi", "--rsmiles", default="", type=str, help="Extract the template for a single reaction SMILES string.")
    parser.add_argument("-sd", "--save-extended-dataset",  action="store_true", help="Save the extended dataset.")
    parser.add_argument("-v", "--verbose",  action="store_true", help="Show messages which occur during processing.")

    args = parser.parse_args()

    return args

def automatic_template_extraction(config_path: str):
    """
    Description:
        Script for automatic reaction template extraction.
    """

    config = Config.load_configuration(config_path)
    template_extraction = TemplateExtraction(config.template_extraction_config)

    return template_extraction.extract_reaction_templates()

def main(args):
    """
    Description:
        Load the configuration file and run the automatic reaction template extraction specified by the user-specified arguments.
    """

    config = Config.load_configuration(args.config)
    template_extraction = TemplateExtraction(config.template_extraction_config)

    if args.rsmiles != "":
        print("\nOriginal Reaction SMILES: {}".format(args.rsmiles))

        if args.task == "clean":
            print("Cleaned Reaction SMILES: {}".format(template_extraction.clean_entry(args.rsmiles, verbose=args.verbose)))
        
        elif args.task == "map":
            print("Atom-mapped Reaction SMILES: {}".format(template_extraction.atom_map_entry(args.rsmiles, verbose=args.verbose)))
        
        elif args.task == "ext_tmplt":
            print("Extracted Template SMARTS: {}".format(template_extraction.extract_reaction_template_from_entry(args.rsmiles, verbose=args.verbose)))
        
        elif args.task == "all":
            cleaned_rsmiles = template_extraction.clean_entry(args.rsmiles, verbose=args.verbose)
            print("Cleaned Reaction SMILES: {}".format(cleaned_rsmiles))
            mapped_rsmiles = template_extraction.atom_map_entry(cleaned_rsmiles, verbose=args.verbose)
            print("Atom-mapped Reaction SMILES: {}".format(mapped_rsmiles))
            print("Extracted Template SMARTS: {}".format(template_extraction.extract_reaction_template_from_entry(mapped_rsmiles, verbose=args.verbose)))
    
    else:
        template_extraction.extract_reaction_templates(save_extended_dataset=args.save_extended_dataset, verbose=args.verbose)

if __name__ == "__main__":
    args = parse_user_args()
    main(args=args)

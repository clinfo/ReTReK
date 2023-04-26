from argparse import ArgumentParser

from config import Config
from reimplemented_libraries.data_pre_processing import DataPreProcessing


def parse_user_args():
    """
    Description:
        Parse the arguments specified by the user during input.
    """

    parser = ArgumentParser("Data pre-processing")

    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("-smi", "--smiles", default="", type=str, help="Process only a single SMILES string.")
    parser.add_argument("-v", "--verbose",  action="store_true", help="Show messages which occur during processing.")

    args = parser.parse_args()

    return args

def main(args):
    """
    Description:
        Load the configuration file and run the pre-processing according to the user-specified arguments.
    """

    config = Config.load_configuration(args.config)
    data_pre_processing = DataPreProcessing(config.data_pre_processing_config)

    # NOTEEEEEEEEEEEEEE: Delete Afterwards !!!
    # data_pre_processing.standardize_zinc15_dataset(zinc15_folder_path="/nasa/datasets/riken_retrosynthesis/zinc15/")
    # exit(0)
    
    if args.smiles != "":
        print("Pre-processed SMILES and SA_Score: {}".format(data_pre_processing.pre_process_entry(args.smiles, verbose=args.verbose)))
    else:
        data_pre_processing.pre_process_data(verbose=args.verbose)

if __name__ == "__main__":
    args = parse_user_args()
    main(args=args)

from rdkit import Chem


def main():
    suppl = Chem.SmilesMolSupplier("ChEMBL_compounds_USAN_YEAR_mt_2017_standardized.smi")
    mols = [m for m in suppl if m is not None]
    mols = [m for m in mols if m.GetNumAtoms() <= 50]
    smiles_list = [Chem.MolToSmiles(m) for m in mols]
    with open("ChEMBL_compounds_USAN_YEAR_mt_2017_standardized_lt_an50.smi", 'w') as f:
        f.write('\n'.join(smiles_list))

if __name__ == "__main__":
    main()

# ReTReK: ReTrosynthesis planning application using Retrosynthesis Knowledge
This package provides a data-driven computer-aided synthesis planning tool using retrosynthesis knowledge. 
In this package, the model of ReTReK was trained with US Patent dataset instead of Reaxys reaction dataset. 

<div align="center">
  <img src="./images/ReTReK_summary.jpg">
</div>

## Dependancy
### Environment (confirmed)
- Ubuntu: 18.04 (model training & synthetic route prediction)
- macOS Catalina: 10.15.7 (synthetic route prediction)

### Package
- python: 3.7
    - tensorflow-gpu: 1.13.1
    - keras-gpu: 2.3.1
    - cudatoolkit: 10.0.130
    - RDKit: 2020.03.2.0
    - [py4j](https://www.py4j.org/): 0.10.8.1
    - tqdm: 4.46.0
    - [kGCN](https://github.com/clinfo/kGCN)
- Java: 1.8
    - [ChemAxon](https://docs.chemaxon.com/display/docs/Installing_to_Servers.html): 20.13
    - [Commons Collections](https://commons.apache.org/proper/commons-collections/download_collections.cgi): 4.4
    - [args4j](https://search.maven.org/search?q=g:args4j%20AND%20a:args4j): 2.33

## Setup
Please refer to the following [link](doc/setup.md). 

## Example usage

Note: The order of the knowledge arguments corresponds to that of the knowledge_weight arguments. 
```bash
javac CxnUtils.java  # for the first time only

# use all knowledge
python run.py --config config/sample.json --target data/sample.mol --knowledge cdscore rdscore asscore stscore --knowledge_weights 1.0 1.0 1.0 1.0

# use CDScore with a weight of 2.0
python run.py --config config/sample.json --target data/sample.mol --knowledge cdscore --knowledge_weights 2.0 0.0 0.0 0.0
```
If you want to try your own molecule, prepare the molecule as MDL MOLfile format and replace `data/sample.mol` with the prepared file. 

# Optional arguments

- `--sel_const`: constant value for selection (default value is set to 3). 
- `--expansion_num`: number of reaction templates used in the expansion step (default value is set to 50). 
- `--starting_material`: path to SMILES format file containing starting materials. 
- `--search_count`: the maximum number of iterations of MCTS (default value is set to 100). 

## Terms
### Convergent Disconnection Score (CDScore)
CDScore aims to favor convergent synthesis, which is known as an efficient strategy in multi-step chemical synthesis. 

### Available Substances Score (ASScore)
For a similar purpose of CDScore, the number of available substances generated in a reaction step is calculated. 

### Ring Disconnection Score (RDScore)
A ring construction strategy is preferred if a target compounds has complex ring structures.

### Selective Transformation Score (STScore)
A synthetic reaction with few by-products is generally preferred in terms of yield.

## Contact
Shoichi Ishida: ishida.shouichi.57a@st.kyoto-u.ac.jp

## Reference

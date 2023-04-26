# Predictive model using kMol

This module offers capabilities to train models to predict ReTReK results.

The predictions includes:
- Probability of being solved by ReTReK
- If solved:
	- Number of steps
	- Minimum, average and maximum number of associated templates in the USPTO for the reaction templates used in the synthetic path found.

## Installing dependencies

This module relies on [kMol](https://github.com/elix-tech/kmol)
 library, please refer to this library for installation instructions.

Once [kMol](https://github.com/elix-tech/kmol) is installed, make sure the environement is activated `conda activate kmol`, then the following scripts can be used.

## Creating dataset for training

The dataset necessary to train the predictive models can be created using the `create_dataset.py` script.

First, download the `uspto_grants_templates.csv` and `elix_retrek_eval_27k_092021.zip` in the following [shared box folder](https://app.box.com/folder/154520948330?s=l0aymv3y6njy6fb3vqe7gz63el2ce4s8).

Unzip the `elix_retrek_eval_27k_092021.zip` file.

The `create_dataset.py` script can be used with the following arguments:

	python create_dataset.py \
		--data-folder # should point to path where elix_retrek_eval_27k_092021.zip was unzipped.
		--uspto-template-path # should point to path of uspto_grants_templates.csv
		--template-column forward_template # indicates column containing the templates in the uspto templates file.
		--output-path # Path where the resulting csv file should be saved.


## Training / Evaluating models

Once the CSV dataset is generated, predictive models can be trained using the `main.py` script.

The `main.py` script can be used with the following arguments:

	python main.py \
		--config # should point to configuration file
		--task train # can be one of `train`, `eval`, `bayesian_opt`
		--num-workers 4 # number of workers to use for dataset loading (default to 4) 
		--eval-output-path results # Optional, only for eval task, overwrites the `output_path` in the json config file.


## Running inference

Once predictive models have been trained, they can be used to run inference on target compounds using the `inference.py` script.

The `inference.py` script can be used with the following arguments:
	
	python inference.py \
		--classification-model # should point to configuration file of solved / unsolved classification model
		--regression-model # Optional, should point to configuration file of regression model
		--data # either a single SMILES string or path to file containing one SMILES per line
		--save-path # Path to where the results should be saved.
		--featurizer # featurizer to use to preprocess the SMILES, should be based on the models used (default is graph)


## Using Docker

First build the Docker image using
```bash
docker build . -t retrek
```

We use `retrek` as the name for the image as an example but it could be any name.

Then create an alias to run commands more conveniently:
```bash
alias retrek_docker='docker run --rm -it --gpus=all --ipc=host --volume="$(pwd)"/:/retrek/ retrek'
```
If you choose a different name than `retrek` for the docker image, be sure to modify it in the alias (only the last word).

(This line can also be added to your `~/.bashrc` to avoid having to re-entering it after each login.)

Finally al the previous scripts can be run simply by replacing`python` with `retrek_docker`, for example:
```bash
retrek_docker create_dataset.py --data-folder elix_retrek_eval_27k_092021 --uspto-template-path uspto_grants_templates.csv --template-column forward_template --output-path retrek_dataset.csv
```

**As only the current working directory will be available to the Docker container, it is important that all necessary files are included in the current working directory (or one of its subdirectories) when running commands through Docker.**
	
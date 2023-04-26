# predictive model module

## Table of contents

- [Usage](#usage)
	- [Dataset preparation](#dataset-preparation)
	- [Training](#training)
	- [Evaluation](#evaluation)
	- [Hyper parameter optimization](#hyper-parameter-optimization)

## Usage

The user needs to prepare a configuration file (json format) to specify the different parameters for model, dataset, optimization...<br/>
A template can be found in `config/`.

The possible parameters to set are the following:
- `model`: (str) Model name, possible choices are: `GCN`, `GAT` or `GIN`.
- `model.py`: (str) Path to model class.
- `normalize_adj_flag`: (bool) Whether or not to normalize the adjacency matrix.
- `num_gcn_layers`: (int) Number of layers to use in the GNN.
- `hidden_dim`: (int) Dimension of hidden features.
- `pred_dim`: (int) Dimension of last layer before classification
- `concat`: (bool) If set to true then each layer's output will be concatenated as final feature representation.
- `max_pool`: (bool) If set to true, a MaxPooling layer will be used in between each graph convolutionnal layers.
- `readout`: (str) Choice of readout function. Possible choices are: `sum`, `mean`, `mean_max`, `sum_max`.
- `residual`: (bool) If set to true, each layer will use residual connections.
- `batchnorm`: (bool) If set to true, each layer will use batch normalization.
- `batch_size`: (int) Batch size.
- `learning_rate`: (float) Learning rate to use for training.
- `dropout`: (float) Dropout parameter.
- `save_path`: (str) Path to saving directory.
- `checkpoint_path`: (str) Path to checkpoint to load before evaluation.
- `save_interval`: (int) Number of epochs that separate each checkpoint saving.
- `epoch`: (int) Number of epochs to train the model for.
- `lr_scheduler`: (str) Choice of learning rate scheduler: can be `cosine` or `step`.
- `step_div`: (float) Coefficient that will divide the learning rate if `lr_scheduler` is set to `step`.
- `div_factor`: (float) Coefficient that will divide the learning rate to get the final learning rate if `lr_scheduler` is set to `cosine`.
- `regression_weight`: (float) Weight of regression loss during training.
- `category_weight`: (float) Weight softmax cross entropy loss during training (for step prediction).
- `patience`: (int) Number of epochs to give as the patience parameter for Early stopping.
- `tune_params`: (list) List of hyper parameters to tune with Bayesian optimization. Refer to [Hyper parameter optimization](#hyper-parameter-optimization)
- `hyper_param_opt`: (bool) If set to true, hyper parameter optimization will be ran.
- `n_trials`: (int) Number of trials to run the hyper parameter optimization for.

### Dataset preparation

The `create_dataset.py` script can be used to create a dataset that can then be used for training / evaluation purposes.<br/>

The process to create a custom dataset is the following:
- Generate RealRetro results on .mol files.
- Use the obtained result folder as well as the directory containing the .mol files as arguments to the `create_dataset.py` script.

The command should be:

	python create_dataset.py -r path/to/results/directory -d path/to/mol/files/directory -n atom_limit -s save_path

Two additional arguments can be passed:
- `--template-mapper`: path to csv file mapping eah reaction template to the number of records it is found in a reference dataset
(default uses ../data/templates_mapped.sma) which corresponds to the mapping between reaction template to number of records in USPTO.
- `--with-edges`: If set to True, per node edges features will be included in the dataset. Per node edges features are computed for each node
as the sum of the edges features it belongs to.

Once the dataset is created, the resulting json file can be used for training or evaluation.

### Training

Once the configuration file is set, simply run:

	python main.py train -c path/to/config.json -d path/to/dataset.json

The dataset will be split into a 80/20 train/validation split.

### Evaluation

Once the configuration file is set, simply run:

	python main.py eval -c path/to/config.json -d path/to/dataset.json

Evaluation results containing all predictions will be saved to the directory indicated by `save_path` in the config.


### Hyper parameter optimization

In the configuration file, set the `hyper_param_opt` to true and set the parameters to tune as a list as the following example:
```
"tune_params": [
    ["float", "learning_rate", 0.0008, 0.008, true],
    ["cat", "num_gcn_layers", [2,3,5]],
    ["cat", "hidden_dim", [32,64,128,256]],
    ["cat", "readout", ["sum", "sum_max"]],
    ["cat", "pred_dim", [64,128,256]],
    ["int", "epoch", 150, 300, 50]
],
```

For categorical hyper parameters, the options should define all possible values.<br/>
For integer hyper parameters the options should define [lower, upper, step].<br/>
For float hyper parameters the options should define [lower, upper, log_scale (if bool) or step (if float)].


Once the configuration file is set, simply run:

	python main.py train -c path/to/config.json -d path/to/dataset.json

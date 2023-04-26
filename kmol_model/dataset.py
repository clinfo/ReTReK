import logging
import itertools
import torch
import pandas as pd
import random
import numpy as np
from functools import partial
from rdkit import Chem
from typing import Any, Callable, Dict, List, Optional, Union
from torch_geometric.data import Data
from torch_geometric.data.dataloader import Collater as TorchGeometricCollater
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from joblib import Parallel, delayed

import kmol.data.featurizers as kf
from kmol.core.helpers import SuperFactory, CacheManager
from kmol.data.resources import DataPoint, Collater, Batch
from kmol.data.featurizers import AbstractFeaturizer, AbstractDescriptorComputer, MordredDescriptorComputer


class NoSetDeviceCollater(Collater):
    def __init__(self):
        super().__init__(device=None)

    def apply(self, batch: List[DataPoint]) -> Batch:

        batch = self._unpack(batch)
        for key, values in batch.inputs.items():
            batch.inputs[key] = self._collater.collate(values)
        return batch

class DescriptorFeaturizer(AbstractFeaturizer):
    def __init__(
            self, inputs: List[str], outputs: List[str], descriptor_calculator: AbstractDescriptorComputer,
            should_cache: bool = False, rewrite: bool = True
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)
        self._descriptor_calculator = descriptor_calculator


    def _process(self, data: str):
         mol = Chem.MolFromSmiles(data)
         molecule_features = self._descriptor_calculator.run(mol)
         return  torch.FloatTensor(molecule_features)

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, target_columns, smiles_column: str = "smiles", featurizer: str= "graph", use_cache: bool = True, num_workers: int = 16, cache_location: str = "/tmp/"):
        super().__init__()
        self.data = pd.read_csv(input_path) if input_path is not None else input_path
        self.input_columns = [smiles_column]
        self.target_columns = target_columns
        self._cache_manager = CacheManager(cache_location=cache_location)
        self.cache = {}
        if featurizer == "graph":
            self.featurizer = kf.GraphFeaturizer(inputs=[smiles_column], outputs=["graph"], rewrite=False, descriptor_calculator=kf.RdkitDescriptorComputer())
        elif featurizer == "ecfp":
            self.featurizer = kf.CircularFingerprintFeaturizer(inputs=[smiles_column], outputs=["features"])
        elif featurizer == "mordred":
            self.featurizer = DescriptorFeaturizer(inputs=[smiles_column], outputs=["features"], descriptor_calculator=MordredDescriptorComputer(), should_cache=True)
        else:
            raise ValueError(f"Unknown featurizer type: {featurizer}. Use one of : 'graph', 'ecfp', 'mordred'.")
        if use_cache:
            logging.info("Caching dataset...")
            self.cache = self._cache_manager.execute_cached_operation(
                processor=self._prepare_cache, clear_cache=False, arguments={"num_workers": num_workers}, cache_key={
                    "input_path": input_path,
                    "target_columns": target_columns,
                    "smiles_column": smiles_column,
                    "featurizer": featurizer
                }
            )

    def _prepare_cache(self, num_workers):
        all_ids = self.list_ids()
        chunk_size = len(all_ids) // num_workers
        chunks = [all_ids[i: i+chunk_size] for i in range(0, len(all_ids), chunk_size)]
        chunks = [Subset(self, chunk) for chunk in chunks]
        dataset = sum(Parallel(n_jobs=len(chunks))(delayed(self._prepare_chunk)(chunk) for chunk in chunks), [])

        return {sample.id_: sample for sample in dataset}

    def _prepare_chunk(self, loader) -> List[DataPoint]:
        dataset = []
        with tqdm(total=len(loader)) as progress_bar:
            for sample in loader:
                try:
                    dataset.append(sample)

                except FeaturizationError as e:
                    logging.warning(e)

                progress_bar.update(1)

        return dataset

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        entry = self.data.iloc[idx]
        sample = DataPoint(
            id_=idx,
            labels=self.target_columns,
            inputs={**entry[self.input_columns]},
            outputs=entry[self.target_columns].to_list()
        )
        self.featurizer.run(sample)
        return sample

    def __len__(self):
        return len(self.data)

    def list_ids(self):
        return np.arange(len(self)).tolist()

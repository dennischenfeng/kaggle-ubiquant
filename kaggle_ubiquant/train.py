"""
Functions to help streamline the training pipeline.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from typing import Iterable, Callable, Dict, Any


Dataset = namedtuple('Dataset', ['train', 'test'])


def generate_dataset(
    n_investments_train: int, 
    n_investments_test: int, 
    n_investments_overlap: int,
    df: pd.DataFrame,
    start_test_time_id: int = 900,
) -> Dataset:
    """
    :param n_investments_overlap: num investments to be overlapping in train and test.
    """
    """
    get sampled investments for train
    get sampled invetments for test, accounting for overlap
    build the datasets
    """
    all_investment_ids = pd.unique(df.investment_id)
    # pool of iids for train requires data before start_test_time_id
    all_investment_ids_train = pd.unique(
        df[df.time_id < start_test_time_id].investment_id
    )

    iid_train = np.random.choice(
        all_investment_ids_train, n_investments_train, replace=False
    )
    iid_overlap = np.random.choice(iid_train, n_investments_overlap, replace=False)

    n_test_remaining = n_investments_test - n_investments_overlap
    iid_not_train = [iid for iid in all_investment_ids if iid not in iid_train]
    iid_test_remaining = np.random.choice(
        iid_not_train, n_test_remaining, replace=False
    )
    iid_test = np.append(iid_overlap, iid_test_remaining)

    train_df = df[df.investment_id.isin(iid_train) & (df.time_id < start_test_time_id)]
    test_df = df[df.investment_id.isin(iid_test) & (df.time_id >= start_test_time_id)]
    return Dataset(train=train_df, test=test_df)



@dataclass
class ModelConfig: 
    """
    Data for model configuration, including hparams
    """
    model_cls: Callable
    model_hparams: Dict[str, Any]
    use_investment_id: bool
    num_lags: int = 1
    lag_default_value: float = 0
    

def generate_model(model_config: ModelConfig):
    """
    Initialize a model, ready to train on data.
    """
    assert model_config.num_lags == 1  # TODO allow more lags


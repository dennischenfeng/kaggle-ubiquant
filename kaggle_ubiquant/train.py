"""
Functions to help streamline the training pipeline.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Iterable, Callable, Dict, Any
from tqdm import tqdm
from xgboost import XGBClassifier


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class DatasetConfig:
    num_lags: int = 1
    lag_default_value: float = 0
    use_investment_id: bool = True


def generate_dataset(
    n_investments_train: int, 
    n_investments_test: int, 
    n_investments_overlap: int,
    df: pd.DataFrame,
    dataset_config: DatasetConfig,
    start_test_time_id: int = 900,
) -> Dataset:
    """
    :param n_investments_overlap: num investments to be overlapping in train and test.
    """
    # TODO: allow more flexibility in config
    assert dataset_config.num_lags == 1

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

    # Apply dataset_config
    # TODO

    return Dataset(train_df, test_df)


def compute_lag1(df: pd.DataFrame) -> np.ndarray:
    """
    Lag_1 features (for time steps without a previous time step, just take last known target)
    """
    assert 'investment_id' in df.columns
    assert 'target' in df.columns

    last_target = defaultdict(lambda: 0)
    result = np.zeros(df.shape[0])

    for i in tqdm(df.index):
        iid = df.loc[i, 'investment_id']
        result[i] = last_target[iid]
        last_target[iid] = df.loc[i, 'target']
    return result


@dataclass
class ModelConfig: 
    """
    Data for model configuration, including hparams
    """
    model_cls: Callable = XGBClassifier
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    

def generate_model(model_config: ModelConfig):
    """
    Initialize a model, ready to train on data.
    """
    return model_config.model_cls(**model_config.model_kwargs)


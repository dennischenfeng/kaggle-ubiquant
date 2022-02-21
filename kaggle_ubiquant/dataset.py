"""
Dataset
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Callable, Dict, Any
from tqdm import tqdm


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class DatasetConfig:
    n_investments_train: int 
    n_investments_test: int 
    n_investments_overlap: int
    start_test_time_id: int = 900
    num_lags: int = 0
    lag_default_value: float = 0
    use_investment_id: bool = True


def generate_dataset(
    dataset_config: DatasetConfig,
    df: pd.DataFrame,
) -> Dataset:
    """
    :param n_investments_overlap: num investments to be overlapping in train and test.
    """
    dc = dataset_config

    # Select the rows for train_df and test_df
    all_investment_ids = pd.unique(df.investment_id)
    # pool of iids for train requires data before start_test_time_id
    all_investment_ids_train = pd.unique(
        df[df.time_id < dc.start_test_time_id].investment_id
    )

    iid_train = np.random.choice(
        all_investment_ids_train, dc.n_investments_train, replace=False
    )
    iid_overlap = np.random.choice(iid_train, dc.n_investments_overlap, replace=False)

    n_test_remaining = dc.n_investments_test - dc.n_investments_overlap
    iid_not_train = [iid for iid in all_investment_ids if iid not in iid_train]
    iid_test_remaining = np.random.choice(
        iid_not_train, n_test_remaining, replace=False
    )
    iid_test = np.append(iid_overlap, iid_test_remaining)

    train_df = df[df.investment_id.isin(iid_train) & (df.time_id < dc.start_test_time_id)]
    test_df = df[df.investment_id.isin(iid_test) & (df.time_id >= dc.start_test_time_id)]

    # Add lag column if needed
    lag_columns = set()
    for df in [train_df, test_df]:
        if dc.num_lags == 1:
            df['target_lag1'] = compute_lag1(df, lag_default_value=dc.lag_default_value)
            lag_columns.add('target_lag1')
        elif dc.num_lags > 1:
            raise NotImplementedError('`num_lags` > 1 is not implemented yet.')

    # Select the columns for train_df and test_df
    df_columns = ['time_id', 'target'] + list(lag_columns)
    if dc.use_investment_id: df_columns.append('investment_id')
    df_columns.extend([f'f_{i}' for i in range(300)])
    train_df = train_df[df_columns]
    test_df = test_df[df_columns]

    return Dataset(train_df, test_df)


def compute_lag1(df: pd.DataFrame, lag_default_value: float = 0) -> np.ndarray:
    """
    Lag_1 features (for time steps without a previous time step, just take last known target)
    """
    assert 'investment_id' in df.columns
    assert 'target' in df.columns

    last_target = defaultdict(lambda: lag_default_value)
    result = df.target.copy()
    result.name = 'target_lag1'

    for i in tqdm(df.index):
        iid = df.loc[i, 'investment_id']
        result[i] = last_target[iid]
        last_target[iid] = df.loc[i, 'target']
    return result

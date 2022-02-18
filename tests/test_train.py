"""
Unit tests for train.py
"""

import pytest
import pandas as pd
import wandb
from xgboost import XGBClassifier
from kaggle_ubiquant.dataset import generate_dataset, compute_lag1, DatasetConfig
from kaggle_ubiquant.model import ModelConfig, generate_model
from kaggle_ubiquant.train import training_run
from definitions import ROOT_DIR
import numpy as np


@pytest.fixture
def df_smallest():
    return pd.read_csv(ROOT_DIR / 'data/train_smallest.csv')


def test_generate_dataset(df_smallest):
    # 1st
    dataset_config = DatasetConfig(5, 3, 2)
    ds = generate_dataset(dataset_config, df_smallest)
    train_iids = pd.unique(ds.train.investment_id)
    assert train_iids.shape[0] == 5
    test_iids = pd.unique(ds.test.investment_id)
    assert test_iids.shape[0] == 3
    overlap_iids = set(train_iids).intersection(set(test_iids))
    assert len(overlap_iids) == 2

    assert 'target_lag1' in ds.train.columns
    assert 'target_lag1' in ds.test.columns
    assert ds.train.iloc[0, ds.train.columns.get_loc('target_lag1')] == 0
    
    # 2nd
    dataset_config = DatasetConfig(5, 3, 2, num_lags=0)
    ds = generate_dataset(dataset_config, df_smallest)
    assert 'target_lag1' not in ds.train

    # 3rd
    dataset_config = DatasetConfig(5, 3, 2, lag_default_value=7.7)
    ds = generate_dataset(dataset_config, df_smallest)
    assert ds.train.iloc[0, ds.train.columns.get_loc('target_lag1')] == 7.7


def test_compute_lag1(df_smallest):
    lag1 = compute_lag1(df_smallest)

    mask_iid2 = (df_smallest.investment_id == 2)
    expected = np.array(df_smallest[mask_iid2]['target'])[:-1]
    actual = lag1[mask_iid2][1:]
    np.testing.assert_allclose(expected, actual)

    mask_iid35 = (df_smallest.investment_id == 35)
    expected = np.array(df_smallest[mask_iid35]['target'])[:-1]
    actual = lag1[mask_iid35][1:]
    np.testing.assert_allclose(expected, actual)


def test_generate_model():
    mc = ModelConfig(model_cls=XGBClassifier, model_kwargs=dict(tree_method='gpu_hist'))
    model = generate_model(mc)
    model.tree_method == 'gpu_hist'


def test_training_run(df_smallest):
    # TODO: how to test robustly with wandb?
    dc = DatasetConfig(7, 5, 5)
    mc = ModelConfig()
    _, r = training_run(df_smallest, dc, mc, wandb_project=None)
    assert -1.0 <= r <= 1.0

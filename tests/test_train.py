"""
Unit tests for train.py
"""

import pytest
import pandas as pd
from xgboost import XGBClassifier
from kaggle_ubiquant.dataset import generate_dataset, compute_lag1, DatasetConfig
from kaggle_ubiquant.model import ModelConfig, generate_model
from definitions import ROOT_DIR
import numpy as np


@pytest.fixture
def df_smallest():
    return pd.read_csv(ROOT_DIR / 'data/train_smallest.csv')


def test_generate_dataset(df_smallest):
    dataset_config = DatasetConfig()
    t = generate_dataset(5, 3, 2, df_smallest, dataset_config)
    df_train = pd.unique(t.train.investment_id)
    assert df_train.shape[0] == 5
    df_test = pd.unique(t.test.investment_id)
    assert df_test.shape[0] == 3
    df_overlap = set(df_train).intersection(set(df_test))
    assert len(df_overlap) == 2


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

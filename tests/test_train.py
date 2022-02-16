"""
Unit tests for train.py
"""

import pytest
import pandas as pd
from kaggle_ubiquant.train import generate_dataset
from definitions import ROOT_DIR


@pytest.fixture
def df_smallest():
    return pd.read_csv(ROOT_DIR / 'data/train_smallest.csv')


def test_generate_dataset(df_smallest):
    t = generate_dataset(5, 3, 2, df_smallest)
    df_train = pd.unique(t.train.investment_id)
    assert df_train.shape[0] == 5
    df_test = pd.unique(t.test.investment_id)
    assert df_test.shape[0] == 3
    df_overlap = set(df_train).intersection(set(df_test))
    assert len(df_overlap) == 2

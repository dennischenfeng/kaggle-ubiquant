"""
Functions to help streamline the training pipeline.
"""

from kaggle_ubiquant.dataset import generate_dataset, DatasetConfig, Dataset
from kaggle_ubiquant.model import generate_model, ModelConfig
import pandas as pd
import numpy as np
from typing import Iterable, Dict, Tuple, Callable, Optional
from scipy.stats import pearsonr
import wandb
import dataclasses


def training_run(
    raw_df: pd.DataFrame,
    dataset_config: DatasetConfig, 
    model_config: ModelConfig, 
    wandb_project: Optional[str] = None,
) -> float:
    """
    Returns a score (goal is to maximize). In this case, it's the Pearson correlation coeff.
    """
    dataset = generate_dataset(dataset_config, raw_df)
    model = generate_model(model_config)
    
    if wandb_project:
        wandb.init(project=wandb_project)
        wandb.config.dataset_config = dataclasses.asdict(dataset_config)
        wandb.config.model_config = dataclasses.asdict(model_config)
        # wandb doesn't unpack more than 1 level down in dictionaries, so need to do manually
        wandb.config['model_config.model_kwargs'] = model_config.model_kwargs
    
    feature_columns = [elem for elem in list(dataset.train.columns) if elem != 'target']
    X = dataset.train.loc[:, feature_columns]
    y = dataset.train.target
    model.fit(X, y)
    
    preds = model.predict(dataset.test.loc[:, feature_columns])
    targets = dataset.test.target
    r, _ = pearsonr(preds, targets)
    
    if wandb_project:
        wandb.log({'pearsonr': r})
        wandb.finish()
    
    return r

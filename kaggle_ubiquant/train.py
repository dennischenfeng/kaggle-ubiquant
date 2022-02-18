"""
Functions to help streamline the training pipeline.
"""

from kaggle_ubiquant.dataset import generate_dataset, DatasetConfig, Dataset
from kaggle_ubiquant.model import generate_model, ModelConfig
import pandas as pd
import numpy as np
from typing import Iterable, Dict, Tuple, Callable, Optional, Any
from scipy.stats import pearsonr
import wandb
import dataclasses


def training_run(
    raw_df: pd.DataFrame,
    dataset_config: DatasetConfig, 
    model_config: ModelConfig, 
    wandb_project: Optional[str] = None,
) -> Tuple[Any, float]:
    """
    Returns the model and the score (the goal is to maximize). In this case, score is the Pearson correlation coeff.
    """
    dataset = generate_dataset(dataset_config, raw_df)
    model = generate_model(model_config)
    
    if wandb_project:
        wandb.init(project=wandb_project)
        wandb.config.dataset_config = dataclasses.asdict(dataset_config)
        wandb.config.model_config = dataclasses.asdict(model_config)
    
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
    
    return model, r

def test():
    print('hi')
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Intro
# Here, we train a XGB model over different hparams/configs

# %%
from kaggle_ubiquant.dataset import generate_dataset, DatasetConfig, Dataset
from kaggle_ubiquant.model import generate_model, ModelConfig
from kaggle_ubiquant.train import training_run
from definitions import ROOT_DIR
import pandas as pd
import numpy as np
from typing import Iterable, Dict, Tuple, Callable, Optional
from scipy.stats import pearsonr
import wandb
import dataclasses

# %%
df = pd.read_csv(ROOT_DIR / 'data/train_smallest.csv')
df.head()

# %% [markdown]
# # Build training run
# uses wandb

# %%
len(pd.unique(df.investment_id))

# %%
dc = DatasetConfig(7, 5, 5)
mc = ModelConfig()

# %% jupyter={"source_hidden": true} tags=[]
# Moved code to git repo

# def training_run(
#     raw_df: pd.DataFrame,
#     dataset_config: DatasetConfig, 
#     model_config: ModelConfig, 
#     wandb_project: Optional[str] = None,
# ) -> float:
#     """
#     Returns a score (goal is to maximize). In this case, it's the Pearson correlation coeff.
#     """
#     dataset = generate_dataset(dataset_config, raw_df)
#     model = generate_model(model_config)
    
#     if wandb_project:
#         wandb.init(project=wandb_project)
#         wandb.config.dataset_config = dataclasses.asdict(dataset_config)
#         wandb.config.model_config = dataclasses.asdict(model_config)
    
#     feature_columns = [elem for elem in list(dataset.train.columns) if elem != 'target']
#     X = dataset.train.loc[:, feature_columns]
#     y = dataset.train.target
#     model.fit(X, y)
    
#     preds = model.predict(dataset.test.loc[:, feature_columns])
#     targets = dataset.test.target
#     r, _ = pearsonr(preds, targets)
    
#     if wandb_project:
#         wandb.log({'pearsonr': r})
#         wandb.finish()
    
#     return r

# %%
training_run(df, dc, mc, 'kaggleUbiquant-scratch')

# %% [markdown]
# # Investigate wandb.sklearn
# Conclusion: only helps with visualizations/plotting, not much else

# %%
X_train = dataset.train.loc[:, feature_columns]
y_train = dataset.train.target
X_test = dataset.test.loc[:, feature_columns]
y_test = dataset.test.target

# %%
wandb.init(project='kaggleUbiquant-scratch')
wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test)

# %%
wandb.finish()

# %%

# %% [markdown]
# # Build training loop
# with optuna to choose hparams

# %%

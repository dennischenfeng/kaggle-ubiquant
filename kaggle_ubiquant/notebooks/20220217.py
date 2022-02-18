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
import optuna

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

# %% [markdown]
# (Moved code and unit test to git repo)

# %% [markdown]
# # Investigate wandb.sklearn

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

# %% [markdown]
# Conclusion: only helps with visualizations/plotting, not much else

# %% [markdown]
# # Build training loop
# with optuna to choose hparams

# %%
study = optuna.create_study(direction='maximize')
for i in range(1):
    trial = study.ask()
    
    # if i == 0:
    #     dc = DatasetConfig(7, 5, 5)
    #     mc = ModelConfig()
    # else:
    dc = DatasetConfig(
        7, 5, 5,
        use_investment_id=trial.suggest_categorical('use_investment_id', [True, False]),
    )
    mc = ModelConfig(
        model_kwargs=dict(
            learning_rate=trial.suggest_categorical(
                'learning_rate',[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
            ),
            max_depth=trial.suggest_categorical(
                'max_depth', [3, 4, 5, 6, 8, 10, 12, 15]
            ),
            min_child_weight=trial.suggest_categorical(
                'min_child_cweight', [1, 3, 5, 7]
            ),
            gamma=trial.suggest_categorical(
                'gamma', [0.0, 0.1, 0.2 , 0.3, 0.4]
            ),
            colsample_bytree=trial.suggest_categorical(
                'colsample_bytree', [ 0.3, 0.4, 0.5 , 0.7 ]
            ),
        )
    )
    
    scores = []
    for j in range(1):
        scores.append(training_run(df, dc, mc, 'kaggleUbiquant-scratch'))
        
    study.tell(trial, np.mean(scores))

# %% [markdown]
# Conclusion: okay, ready to use Google Colab to run the sweep now

# %%

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

# %%
import numpy as np
from definitions import ROOT_DIR
import pandas as pd
from typing import Iterable, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict
import plotly.express as px

# %%
root_path = ROOT_DIR

# %% [markdown]
# # Datasets

# %%
# To do quick tests, use a smaller section of the data. 
df = pd.read_csv(f'{root_path}/data/train_small2.csv')

# %%
# first ensure time_id are always increasing or staying same
prev_time_id = 0
for _, row in tqdm(df.iterrows()):
    assert row.time_id >= prev_time_id
    prev_time_id = row.time_id

# %%
# include lag_1 features (for time steps without a 
# previous time step, just take last known target)
last_target = defaultdict(lambda: 0)
df['target_lag1'] = 0

for i in tqdm(df.index):
    iid = df.loc[i, 'investment_id']
    df.loc[i, 'target_lag1'] = last_target[iid]
    last_target[iid] = df.loc[i, 'target']

# %%
test_proportion = 0.10
num_test = int(test_proportion * df.shape[0])
train_df = df.iloc[:-num_test, :]
test_df = df.iloc[-num_test:, :]

# %%
train_df.shape, test_df.shape

# %%
plots = []
for iid in df.investment_id[:2]:
    plot = px.line(
        x=df[df.investment_id == iid].time_id,
        y=df[df.investment_id == iid].target
    )
    plot.show()
    plots.append(plot)

# %% [markdown]
# # Scratch

# %%
df_smallest = df[df.investment_id.isin(pd.unique(df.investment_id)[:10])]

# %%
df_smallest.columns

# %%

# %%
df_smallest = df_smallest.drop(columns=['Unnamed: 0'])

# %%
df_smallest

# %%
df_smallest.to_csv('train_smallest.csv')

# %%

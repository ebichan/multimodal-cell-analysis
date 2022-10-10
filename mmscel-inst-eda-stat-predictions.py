# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] papermill={"duration": 0.007743, "end_time": "2022-08-18T12:07:31.764941", "exception": false, "start_time": "2022-08-18T12:07:31.757198", "status": "completed"} tags=[]
# # Multimodal Single-Cell🧬IIntegration: EDA 🔍 & simple predictions

# + _kg_hide-output=true papermill={"duration": 12.182013, "end_time": "2022-08-18T12:07:43.954087", "exception": false, "start_time": "2022-08-18T12:07:31.772074", "status": "completed"} tags=[]
# ! pip install -q tables

# + papermill={"duration": 2.598493, "end_time": "2022-08-18T12:07:46.559481", "exception": false, "start_time": "2022-08-18T12:07:43.960988", "status": "completed"} tags=[]
# %matplotlib inline

import os
import torch
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

PATH_DATASET = "/kaggle/input/open-problems-multimodal"

# + [markdown] papermill={"duration": 0.007001, "end_time": "2022-08-18T12:07:46.573765", "exception": false, "start_time": "2022-08-18T12:07:46.566764", "status": "completed"} tags=[]
# # Browsing the matadata

# + papermill={"duration": 0.336492, "end_time": "2022-08-18T12:07:46.917420", "exception": false, "start_time": "2022-08-18T12:07:46.580928", "status": "completed"} tags=[]
df_meta = pd.read_csv(os.path.join(PATH_DATASET, "metadata.csv")).set_index("cell_id")
display(df_meta.head())

print(f"table size: {len(df_meta)}")

# + papermill={"duration": 0.317711, "end_time": "2022-08-18T12:07:47.243510", "exception": false, "start_time": "2022-08-18T12:07:46.925799", "status": "completed"} tags=[]
fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
for i, col in enumerate(["day", "donor", "technology"]):
    _= df_meta[[col]].value_counts().plot.pie(ax=axarr[i], autopct='%1.1f%%', ylabel=col)

# + papermill={"duration": 0.587013, "end_time": "2022-08-18T12:07:47.841481", "exception": false, "start_time": "2022-08-18T12:07:47.254468", "status": "completed"} tags=[]
fig, axarr = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
for i, col in enumerate(["cell_type", "day", "technology"]):
    _= df_meta.groupby([col, 'donor']).size().unstack().plot(ax=axarr[i], kind='bar', stacked=True, grid=True)

# + papermill={"duration": 0.717579, "end_time": "2022-08-18T12:07:48.567783", "exception": false, "start_time": "2022-08-18T12:07:47.850204", "status": "completed"} tags=[]
fig, axarr = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
for i, col in enumerate(["cell_type", "donor", "technology"]):
    _= df_meta.groupby([col, 'day']).size().unstack().plot(ax=axarr[i], kind='bar', stacked=True, grid=True)

# + [markdown] papermill={"duration": 0.00869, "end_time": "2022-08-18T12:07:48.585680", "exception": false, "start_time": "2022-08-18T12:07:48.576990", "status": "completed"} tags=[]
# # Browse the train dataset

# + papermill={"duration": 59.055777, "end_time": "2022-08-18T12:08:47.650308", "exception": false, "start_time": "2022-08-18T12:07:48.594531", "status": "completed"} tags=[]
df_cite = pd.read_hdf(os.path.join(PATH_DATASET, "train_cite_inputs.h5")).astype(np.float16)
cols_source = list(df_cite.columns)
display(df_cite.head())

# + papermill={"duration": 0.811481, "end_time": "2022-08-18T12:08:48.470279", "exception": false, "start_time": "2022-08-18T12:08:47.658798", "status": "completed"} tags=[]
df = pd.read_hdf(os.path.join(PATH_DATASET, "train_cite_targets.h5")).astype(np.float16)
cols_target = list(df.columns)
display(df.head())

# + papermill={"duration": 11.114146, "end_time": "2022-08-18T12:08:59.593762", "exception": false, "start_time": "2022-08-18T12:08:48.479616", "status": "completed"} tags=[]
df_cite = df_cite.join(df, how='right')
df_cite = df_cite.join(df_meta, how="left")
del df

print(f"total: {len(df_cite)}")
print(f"cell_id: {len(df_cite)}")
display(df_cite.head())

# + [markdown] papermill={"duration": 0.009597, "end_time": "2022-08-18T12:08:59.613285", "exception": false, "start_time": "2022-08-18T12:08:59.603688", "status": "completed"} tags=[]
# ### Meta-data details

# + papermill={"duration": 1.192078, "end_time": "2022-08-18T12:09:00.814992", "exception": false, "start_time": "2022-08-18T12:08:59.622914", "status": "completed"} tags=[]
fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, col in enumerate(["day", "donor", "cell_type"]):
    _= df_cite[[col]].value_counts().plot.pie(ax=axarr[i], autopct='%1.1f%%', ylabel=col)

# + papermill={"duration": 0.317066, "end_time": "2022-08-18T12:09:01.146426", "exception": false, "start_time": "2022-08-18T12:09:00.829360", "status": "completed"} tags=[]
fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
for i, col in enumerate(["day", "cell_type"]):
    _= df_cite.groupby([col, 'donor']).size().unstack().plot(ax=axarr[i], kind='bar', stacked=True, grid=True)

# + papermill={"duration": 0.024528, "end_time": "2022-08-18T12:09:01.181253", "exception": false, "start_time": "2022-08-18T12:09:01.156725", "status": "completed"} tags=[]
del df_cite

# + [markdown] papermill={"duration": 0.01, "end_time": "2022-08-18T12:09:01.201823", "exception": false, "start_time": "2022-08-18T12:09:01.191823", "status": "completed"} tags=[]
# ## Just a fraction of Multi dataset
#
# Note that this dataset is too large to be leaded directly in DataFrame and crashes on Kaggle kernel

# + papermill={"duration": 0.964824, "end_time": "2022-08-18T12:09:02.176883", "exception": false, "start_time": "2022-08-18T12:09:01.212059", "status": "completed"} tags=[]
display(pd.read_hdf(os.path.join(PATH_DATASET, "train_multi_inputs.h5"), start=0, stop=100).head())

# + papermill={"duration": 0.193709, "end_time": "2022-08-18T12:09:02.381838", "exception": false, "start_time": "2022-08-18T12:09:02.188129", "status": "completed"} tags=[]
display(pd.read_hdf(os.path.join(PATH_DATASET, "train_multi_targets.h5"), start=0, stop=100).head())

# + [markdown] papermill={"duration": 0.011918, "end_time": "2022-08-18T12:09:02.404943", "exception": false, "start_time": "2022-08-18T12:09:02.393025", "status": "completed"} tags=[]
# ### Load all indexes from chunks

# + _kg_hide-output=true papermill={"duration": 66.212244, "end_time": "2022-08-18T12:10:08.628228", "exception": false, "start_time": "2022-08-18T12:09:02.415984", "status": "completed"} tags=[]
cell_id = []
for i in range(20):
    df = pd.read_hdf(os.path.join(PATH_DATASET, "train_multi_targets.h5"), start=i * 10000, stop=(i+1) * 10000)
    print(i, len(df), df["ENSG00000121410"].mean())
    if len(df) == 0:
        break
    cell_id += list(df.index)

df_multi_ = pd.DataFrame({"cell_id": cell_id}).set_index("cell_id")
display(df_multi_.head())

# + papermill={"duration": 0.081661, "end_time": "2022-08-18T12:10:08.722018", "exception": false, "start_time": "2022-08-18T12:10:08.640357", "status": "completed"} tags=[]
df_multi_ = df_multi_.join(df_meta, how="left")

print(f"total: {len(df_multi_)}")
print(f"cell_id: {len(df_multi_)}")
display(df_multi_.head())

# + [markdown] papermill={"duration": 0.011845, "end_time": "2022-08-18T12:10:08.746013", "exception": false, "start_time": "2022-08-18T12:10:08.734168", "status": "completed"} tags=[]
# ### Meta-data details

# + papermill={"duration": 0.252565, "end_time": "2022-08-18T12:10:09.010648", "exception": false, "start_time": "2022-08-18T12:10:08.758083", "status": "completed"} tags=[]
fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, col in enumerate(["day", "donor", "cell_type"]):
    _= df_multi_[[col]].value_counts().plot.pie(ax=axarr[i], autopct='%1.1f%%', ylabel=col)

# + papermill={"duration": 0.342582, "end_time": "2022-08-18T12:10:09.371342", "exception": false, "start_time": "2022-08-18T12:10:09.028760", "status": "completed"} tags=[]
fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
for i, col in enumerate(["day", "cell_type"]):
    _= df_multi_.groupby([col, 'donor']).size().unstack().plot(ax=axarr[i], kind='bar', stacked=True, grid=True)

# + [markdown] papermill={"duration": 0.013158, "end_time": "2022-08-18T12:10:09.397739", "exception": false, "start_time": "2022-08-18T12:10:09.384581", "status": "completed"} tags=[]
# # Show Evaluation table

# + papermill={"duration": 58.205088, "end_time": "2022-08-18T12:11:07.616184", "exception": false, "start_time": "2022-08-18T12:10:09.411096", "status": "completed"} tags=[]
df_eval = pd.read_csv(os.path.join(PATH_DATASET, "evaluation_ids.csv")).set_index("row_id")
display(df_eval.head())

print(f"total: {len(df_eval)}")
print(f"cell_id: {len(df_eval['cell_id'].unique())}")
print(f"gene_id: {len(df_eval['gene_id'].unique())}")

# + [markdown] papermill={"duration": 0.013526, "end_time": "2022-08-18T12:11:07.643546", "exception": false, "start_time": "2022-08-18T12:11:07.630020", "status": "completed"} tags=[]
# **NOTE** that this evaluation expect you to run predictions on the both datasets: cite & multi (as you can see bellow)
#
# target columns for:
# - **cite**: 140 columns
# - **multi**: 23418 columns

# + papermill={"duration": 0.368315, "end_time": "2022-08-18T12:11:08.026802", "exception": false, "start_time": "2022-08-18T12:11:07.658487", "status": "completed"} tags=[]
# ! head ../input/open-problems-multimodal/sample_submission.csv

# + [markdown] papermill={"duration": 0.014241, "end_time": "2022-08-18T12:11:08.056020", "exception": false, "start_time": "2022-08-18T12:11:08.041779", "status": "completed"} tags=[]
# # Statistic predictions 🏴‍ gene means

# + papermill={"duration": 0.707506, "end_time": "2022-08-18T12:11:08.777510", "exception": false, "start_time": "2022-08-18T12:11:08.070004", "status": "completed"} tags=[]
col_means = dict(pd.read_hdf(os.path.join(PATH_DATASET, "train_cite_targets.h5")).mean())

# + _kg_hide-output=true papermill={"duration": 71.979079, "end_time": "2022-08-18T12:12:20.770725", "exception": false, "start_time": "2022-08-18T12:11:08.791646", "status": "completed"} tags=[]
col_sums = []
count = 0
for i in range(20):
    df = pd.read_hdf(os.path.join(PATH_DATASET, "train_multi_targets.h5"), start=i * 10000, stop=(i+1) * 10000)
    count += len(df)
    if len(df) == 0:
        break
    col_sums.append(dict(df.sum()))

df_multi_ = pd.DataFrame(col_sums)
display(df_multi_)

# + papermill={"duration": 0.154323, "end_time": "2022-08-18T12:12:20.939283", "exception": false, "start_time": "2022-08-18T12:12:20.784960", "status": "completed"} tags=[]
col_means.update(dict(df_multi_.sum() / count))

# + [markdown] papermill={"duration": 0.014084, "end_time": "2022-08-18T12:12:20.967781", "exception": false, "start_time": "2022-08-18T12:12:20.953697", "status": "completed"} tags=[]
# ## Map target to eval. table

# + papermill={"duration": 7.851821, "end_time": "2022-08-18T12:12:28.833706", "exception": false, "start_time": "2022-08-18T12:12:20.981885", "status": "completed"} tags=[]
df_eval["target"] = df_eval["gene_id"].map(col_means)
display(df_eval)

# + [markdown] papermill={"duration": 0.014547, "end_time": "2022-08-18T12:12:28.863253", "exception": false, "start_time": "2022-08-18T12:12:28.848706", "status": "completed"} tags=[]
# ## Finalize submission

# + papermill={"duration": 108.331169, "end_time": "2022-08-18T12:14:17.209216", "exception": false, "start_time": "2022-08-18T12:12:28.878047", "status": "completed"} tags=[]
df_eval[["target"]].round(6).to_csv("submission.csv")

# ! ls -lh .
# ! head submission.csv
